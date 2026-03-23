use serde::Deserialize;
use serde_json::json;
use std::fmt;

const INFO_LEVEL: u32 = 2;
const ERROR_LEVEL: u32 = 4;
const DEFAULT_PROMPT: &str = "Describe this image in detail";
// Vision responses can include detailed descriptions and large JSON envelopes from the host bridge.
const MAX_HOST_STRING_LEN: usize = 65_536;
const ANTHROPIC_URL: &str = "https://api.anthropic.com/v1/messages";
const OPENAI_URL: &str = "https://api.openai.com/v1/chat/completions";

#[link(wasm_import_module = "host_api_v1")]
extern "C" {
    #[link_name = "log"]
    fn host_log(level: u32, msg_ptr: *const u8, msg_len: u32);
    #[link_name = "get_input"]
    fn host_get_input() -> u32;
    #[link_name = "set_output"]
    fn host_set_output(text_ptr: *const u8, text_len: u32);
    #[link_name = "kv_get"]
    fn host_kv_get(key_ptr: *const u8, key_len: u32) -> u32;
    #[link_name = "http_request"]
    fn host_http_request(
        method_ptr: *const u8,
        method_len: u32,
        url_ptr: *const u8,
        url_len: u32,
        headers_ptr: *const u8,
        headers_len: u32,
        body_ptr: *const u8,
        body_len: u32,
    ) -> u32;
}

#[derive(Debug, Deserialize)]
struct VisionInput {
    image: String,
    prompt: Option<String>,
    provider: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RequestOptions {
    image: ImageInput,
    prompt: String,
    provider: Provider,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ImageInput {
    Url(String),
    DataUri(DataImage),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DataImage {
    media_type: String,
    base64_data: String,
    data_uri: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Provider {
    Anthropic,
    OpenAi,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ApiRequest {
    url: String,
    headers: String,
    body: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum VisionError {
    InvalidInput(String),
    MissingApiKey(String),
    RequestFailed(String),
    ParseFailed(String),
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAiMessage {
    content: OpenAiContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OpenAiContent {
    Text(String),
    Parts(Vec<OpenAiContentPart>),
}

#[derive(Debug, Deserialize)]
struct OpenAiContentPart {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

struct HttpRequest<'a> {
    method: &'a str,
    url: &'a str,
    headers: &'a str,
    body: &'a str,
}

impl fmt::Display for VisionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message)
            | Self::MissingApiKey(message)
            | Self::RequestFailed(message)
            | Self::ParseFailed(message) => formatter.write_str(message),
        }
    }
}

impl Provider {
    fn api_key_name(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic_api_key",
            Self::OpenAi => "openai_api_key",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Anthropic => "Claude",
            Self::OpenAi => "GPT-4o",
        }
    }

    fn missing_key_message(self) -> String {
        format!(
            "No API key found. Set '{}' in skill storage.",
            self.api_key_name()
        )
    }
}

/// # Safety
/// `ptr` must be 0 or point to a NUL-terminated string in valid WASM linear memory.
/// The caller must ensure reading up to `MAX_HOST_STRING_LEN` bytes from `ptr` is safe.
unsafe fn read_host_string(ptr: u32) -> Option<String> {
    if ptr == 0 {
        return None;
    }

    let slice = core::slice::from_raw_parts(ptr as *const u8, MAX_HOST_STRING_LEN);
    let len = slice
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(MAX_HOST_STRING_LEN);
    Some(String::from_utf8_lossy(&slice[..len]).into_owned())
}

fn log(level: u32, message: &str) {
    unsafe {
        host_log(level, message.as_ptr(), message.len() as u32);
    }
}

fn get_input() -> String {
    unsafe { read_host_string(host_get_input()).unwrap_or_default() }
}

fn set_output(text: &str) {
    unsafe {
        host_set_output(text.as_ptr(), text.len() as u32);
    }
}

fn kv_get(key: &str) -> Option<String> {
    unsafe { read_host_string(host_kv_get(key.as_ptr(), key.len() as u32)) }
}

fn http_request(request: &HttpRequest<'_>) -> Option<String> {
    unsafe {
        read_host_string(host_http_request(
            request.method.as_ptr(),
            request.method.len() as u32,
            request.url.as_ptr(),
            request.url.len() as u32,
            request.headers.as_ptr(),
            request.headers.len() as u32,
            request.body.as_ptr(),
            request.body.len() as u32,
        ))
    }
}

fn execute(raw_input: &str) -> Result<String, VisionError> {
    let options = parse_input(raw_input)?;
    let api_key = get_api_key(options.provider)?;
    let request = build_api_request(&options, &api_key);
    let response = send_request(&request)?;
    let analysis = parse_response(options.provider, &response)?;
    Ok(format_output(options.provider, &analysis))
}

fn parse_input(raw_input: &str) -> Result<RequestOptions, VisionError> {
    let input: VisionInput = serde_json::from_str(raw_input)
        .map_err(|error| VisionError::InvalidInput(format!("Invalid input JSON: {error}")))?;

    Ok(RequestOptions {
        image: parse_image(&input.image)?,
        prompt: normalize_prompt(input.prompt),
        provider: parse_provider(input.provider.as_deref())?,
    })
}

fn normalize_prompt(prompt: Option<String>) -> String {
    prompt
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_PROMPT.to_string())
}

fn parse_provider(provider: Option<&str>) -> Result<Provider, VisionError> {
    match provider.map(str::trim).filter(|value| !value.is_empty()) {
        None => Ok(Provider::Anthropic),
        Some(value) if value.eq_ignore_ascii_case("anthropic") => Ok(Provider::Anthropic),
        Some(value) if value.eq_ignore_ascii_case("openai") => Ok(Provider::OpenAi),
        Some(value) => Err(VisionError::InvalidInput(format!(
            "Unknown provider '{value}'. Use 'anthropic' or 'openai'."
        ))),
    }
}

fn parse_image(image: &str) -> Result<ImageInput, VisionError> {
    let trimmed = image.trim();
    if trimmed.starts_with("https://") {
        return Ok(ImageInput::Url(trimmed.to_string()));
    }
    if trimmed.starts_with("data:") {
        return parse_data_image(trimmed).map(ImageInput::DataUri);
    }
    Err(invalid_image_error())
}

fn parse_data_image(data_uri: &str) -> Result<DataImage, VisionError> {
    let (header, base64_data) = data_uri.split_once(',').ok_or_else(invalid_image_error)?;
    let media_type = header
        .strip_prefix("data:")
        .and_then(|value| value.strip_suffix(";base64"))
        .ok_or_else(invalid_image_error)?;

    if base64_data.is_empty() || !is_supported_media_type(media_type) {
        return Err(invalid_image_error());
    }

    Ok(DataImage {
        media_type: media_type.to_string(),
        base64_data: base64_data.to_string(),
        data_uri: data_uri.to_string(),
    })
}

fn is_supported_media_type(media_type: &str) -> bool {
    matches!(
        media_type,
        "image/png" | "image/jpeg" | "image/gif" | "image/webp"
    )
}

fn invalid_image_error() -> VisionError {
    VisionError::InvalidInput(
        "Image must be a URL (https://...) or base64 data URI (data:image/...;base64,...)"
            .to_string(),
    )
}

fn get_api_key(provider: Provider) -> Result<String, VisionError> {
    load_api_key(provider, kv_get(provider.api_key_name()))
}

fn load_api_key(provider: Provider, value: Option<String>) -> Result<String, VisionError> {
    value
        .map(|api_key| api_key.trim().to_string())
        .filter(|api_key| !api_key.is_empty())
        .ok_or_else(|| VisionError::MissingApiKey(provider.missing_key_message()))
}

fn build_api_request(options: &RequestOptions, api_key: &str) -> ApiRequest {
    match options.provider {
        Provider::Anthropic => build_anthropic_request(&options.image, &options.prompt, api_key),
        Provider::OpenAi => build_openai_request(&options.image, &options.prompt, api_key),
    }
}

fn build_anthropic_request(image: &ImageInput, prompt: &str, api_key: &str) -> ApiRequest {
    let headers = json!({
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    })
    .to_string();

    let body = json!({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1024,
        "messages": [{
            "role": "user",
            "content": [anthropic_image_content(image), text_content(prompt)]
        }]
    })
    .to_string();

    ApiRequest {
        url: ANTHROPIC_URL.to_string(),
        headers,
        body,
    }
}

fn build_openai_request(image: &ImageInput, prompt: &str, api_key: &str) -> ApiRequest {
    let headers = json!({
        "Authorization": format!("Bearer {api_key}"),
        "Content-Type": "application/json"
    })
    .to_string();

    let body = json!({
        "model": "gpt-4o-mini",
        "max_tokens": 1024,
        "messages": [{
            "role": "user",
            "content": [openai_image_content(image), text_content(prompt)]
        }]
    })
    .to_string();

    ApiRequest {
        url: OPENAI_URL.to_string(),
        headers,
        body,
    }
}

fn anthropic_image_content(image: &ImageInput) -> serde_json::Value {
    match image {
        ImageInput::Url(url) => json!({
            "type": "image",
            "source": { "type": "url", "url": url }
        }),
        ImageInput::DataUri(data) => json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": data.media_type,
                "data": data.base64_data
            }
        }),
    }
}

fn openai_image_content(image: &ImageInput) -> serde_json::Value {
    let url = match image {
        ImageInput::Url(value) => value.clone(),
        ImageInput::DataUri(data) => data.data_uri.clone(),
    };

    json!({ "type": "image_url", "image_url": { "url": url } })
}

fn text_content(prompt: &str) -> serde_json::Value {
    json!({ "type": "text", "text": prompt })
}

fn send_request(request: &ApiRequest) -> Result<String, VisionError> {
    let call = HttpRequest {
        method: "POST",
        url: &request.url,
        headers: &request.headers,
        body: &request.body,
    };

    require_host_response(http_request(&call))
}

fn require_host_response(response: Option<String>) -> Result<String, VisionError> {
    response
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            VisionError::RequestFailed(
                "Vision API transport failed: no response from host".to_string(),
            )
        })
}

fn parse_response(provider: Provider, response: &str) -> Result<String, VisionError> {
    if is_error_response(response) {
        return Err(VisionError::RequestFailed(error_response_message(response)));
    }

    match provider {
        Provider::Anthropic => parse_anthropic_response(response),
        Provider::OpenAi => parse_openai_response(response),
    }
}

fn error_response_message(response: &str) -> String {
    extract_api_error(response)
        .map(|message| format!("Vision API request failed: {message}"))
        .unwrap_or_else(|| "Vision API request failed: API returned an error response".to_string())
}

fn extract_api_error(response: &str) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(response).ok()?;
    let error = value.get("error")?;

    error
        .get("message")
        .and_then(|message| message.as_str())
        .map(str::to_owned)
        .or_else(|| error.as_str().map(str::to_owned))
}

fn is_error_response(response: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(response) else {
        return false;
    };

    value.get("error").is_some()
        || value
            .get("type")
            .and_then(|kind| kind.as_str())
            .map(|kind| kind == "error")
            .unwrap_or(false)
}

fn parse_anthropic_response(response: &str) -> Result<String, VisionError> {
    let parsed: AnthropicResponse =
        serde_json::from_str(response).map_err(|_| vision_parse_error())?;
    let text = parsed
        .content
        .into_iter()
        .filter_map(anthropic_text_segment)
        .collect();
    join_text_segments(text)
}

fn anthropic_text_segment(content: AnthropicContent) -> Option<String> {
    if content.kind == "text" {
        return content.text;
    }
    None
}

fn parse_openai_response(response: &str) -> Result<String, VisionError> {
    let parsed: OpenAiResponse =
        serde_json::from_str(response).map_err(|_| vision_parse_error())?;
    let choice = parsed
        .choices
        .into_iter()
        .next()
        .ok_or_else(vision_parse_error)?;

    match choice.message.content {
        OpenAiContent::Text(text) => Ok(text),
        OpenAiContent::Parts(parts) => {
            join_text_segments(parts.into_iter().filter_map(openai_text_segment).collect())
        }
    }
}

fn openai_text_segment(part: OpenAiContentPart) -> Option<String> {
    if part.kind == "text" {
        return part.text;
    }
    None
}

fn join_text_segments(segments: Vec<String>) -> Result<String, VisionError> {
    if segments.is_empty() {
        return Err(vision_parse_error());
    }
    Ok(segments.join("\n"))
}

fn vision_parse_error() -> VisionError {
    VisionError::ParseFailed("Failed to parse vision API response".to_string())
}

fn format_output(provider: Provider, analysis: &str) -> String {
    format!("🔍 Image Analysis ({}):\n\n{}", provider.label(), analysis)
}

fn error_output(error: &VisionError) -> String {
    json!({ "error": error.to_string() }).to_string()
}

#[no_mangle]
pub extern "C" fn run() {
    log(INFO_LEVEL, "Vision skill starting");
    let input = get_input();

    match execute(&input) {
        Ok(output) => set_output(&output),
        Err(error) => {
            log(ERROR_LEVEL, &error.to_string());
            set_output(&error_output(&error));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_input_defaults_prompt_and_provider() {
        let options =
            parse_input(r#"{"image":"https://example.com/cat.png"}"#).expect("input should parse");

        assert_eq!(options.prompt, DEFAULT_PROMPT);
        assert_eq!(options.provider, Provider::Anthropic);
        assert_eq!(
            options.image,
            ImageInput::Url("https://example.com/cat.png".to_string())
        );
    }

    #[test]
    fn parse_input_accepts_explicit_prompt_and_provider() {
        let options = parse_input(
            r#"{"image":"https://example.com/cat.png","prompt":"Count the cats","provider":"openai"}"#,
        )
        .expect("input should parse");

        assert_eq!(options.prompt, "Count the cats");
        assert_eq!(options.provider, Provider::OpenAi);
    }

    #[test]
    fn parse_image_detects_https_url() {
        let image = parse_image("https://example.com/image.webp").expect("url should parse");
        assert_eq!(
            image,
            ImageInput::Url("https://example.com/image.webp".to_string())
        );
    }

    #[test]
    fn parse_image_detects_base64_data_uri() {
        let image = parse_image("data:image/png;base64,Zm9v").expect("data uri should parse");
        assert_eq!(
            image,
            ImageInput::DataUri(DataImage {
                media_type: "image/png".to_string(),
                base64_data: "Zm9v".to_string(),
                data_uri: "data:image/png;base64,Zm9v".to_string(),
            })
        );
    }

    #[test]
    fn parse_image_rejects_invalid_format() {
        let error = parse_image("/tmp/cat.png").expect_err("image should be rejected");
        assert_eq!(
            error.to_string(),
            "Image must be a URL (https://...) or base64 data URI (data:image/...;base64,...)"
        );
    }

    #[test]
    fn parse_data_image_extracts_supported_media_types() {
        for media_type in ["image/png", "image/jpeg", "image/gif", "image/webp"] {
            let data_uri = format!("data:{media_type};base64,Zm9v");
            let image = parse_data_image(&data_uri).expect("media type should parse");
            assert_eq!(image.media_type, media_type);
            assert_eq!(image.base64_data, "Zm9v");
        }
    }

    #[test]
    fn build_anthropic_request_uses_url_images() {
        let request = build_anthropic_request(
            &ImageInput::Url("https://example.com/cat.png".to_string()),
            "Describe the cat",
            "secret",
        );
        let headers: serde_json::Value = serde_json::from_str(&request.headers).expect("headers");
        let body: serde_json::Value = serde_json::from_str(&request.body).expect("body");

        assert_eq!(request.url, ANTHROPIC_URL);
        assert_eq!(headers["x-api-key"], "secret");
        assert_eq!(body["messages"][0]["content"][0]["source"]["type"], "url");
        assert_eq!(
            body["messages"][0]["content"][0]["source"]["url"],
            "https://example.com/cat.png"
        );
    }

    #[test]
    fn build_anthropic_request_uses_base64_images() {
        let request = build_anthropic_request(
            &ImageInput::DataUri(DataImage {
                media_type: "image/jpeg".to_string(),
                base64_data: "Zm9v".to_string(),
                data_uri: "data:image/jpeg;base64,Zm9v".to_string(),
            }),
            "Describe the cat",
            "secret",
        );
        let body: serde_json::Value = serde_json::from_str(&request.body).expect("body");

        assert_eq!(
            body["messages"][0]["content"][0]["source"]["type"],
            "base64"
        );
        assert_eq!(
            body["messages"][0]["content"][0]["source"]["media_type"],
            "image/jpeg"
        );
        assert_eq!(body["messages"][0]["content"][0]["source"]["data"], "Zm9v");
    }

    #[test]
    fn build_openai_request_uses_url_images() {
        let request = build_openai_request(
            &ImageInput::Url("https://example.com/cat.png".to_string()),
            "Describe the cat",
            "secret",
        );
        let headers: serde_json::Value = serde_json::from_str(&request.headers).expect("headers");
        let body: serde_json::Value = serde_json::from_str(&request.body).expect("body");

        assert_eq!(request.url, OPENAI_URL);
        assert_eq!(headers["Authorization"], "Bearer secret");
        assert_eq!(
            body["messages"][0]["content"][0]["image_url"]["url"],
            "https://example.com/cat.png"
        );
    }

    #[test]
    fn build_openai_request_uses_data_urls() {
        let request = build_openai_request(
            &ImageInput::DataUri(DataImage {
                media_type: "image/png".to_string(),
                base64_data: "Zm9v".to_string(),
                data_uri: "data:image/png;base64,Zm9v".to_string(),
            }),
            "Describe the cat",
            "secret",
        );
        let body: serde_json::Value = serde_json::from_str(&request.body).expect("body");

        assert_eq!(
            body["messages"][0]["content"][0]["image_url"]["url"],
            "data:image/png;base64,Zm9v"
        );
    }

    #[test]
    fn load_api_key_rejects_missing_keys() {
        let error = load_api_key(Provider::Anthropic, None).expect_err("key should be required");
        assert_eq!(
            error.to_string(),
            "No API key found. Set 'anthropic_api_key' in skill storage."
        );
    }

    #[test]
    fn parse_provider_rejects_unknown_provider() {
        let error = parse_provider(Some("gemini")).expect_err("provider should fail");
        assert_eq!(
            error.to_string(),
            "Unknown provider 'gemini'. Use 'anthropic' or 'openai'."
        );
    }

    #[test]
    fn parse_anthropic_response_extracts_text() {
        let response = r#"{
            "content": [
                {"type": "text", "text": "A black cat sits on a couch."}
            ]
        }"#;

        assert_eq!(
            parse_anthropic_response(response).expect("response should parse"),
            "A black cat sits on a couch."
        );
    }

    #[test]
    fn parse_openai_response_extracts_text() {
        let response = r#"{
            "choices": [
                {"message": {"content": "A black cat sits on a couch."}}
            ]
        }"#;

        assert_eq!(
            parse_openai_response(response).expect("response should parse"),
            "A black cat sits on a couch."
        );
    }

    #[test]
    fn manifest_declares_storage_capability() {
        let manifest = include_str!("../manifest.toml");
        assert!(manifest.contains(r#"capabilities = ["network", "storage"]"#));
    }

    #[test]
    fn require_host_response_reports_transport_failures() {
        let error = require_host_response(None).expect_err("missing host response should fail");
        assert_eq!(
            error.to_string(),
            "Vision API transport failed: no response from host"
        );
    }

    #[test]
    fn parse_response_includes_api_error_details() {
        let response = r#"{"error":{"message":"bad request"}}"#;
        let error = parse_response(Provider::OpenAi, response).expect_err("error expected");
        assert_eq!(error.to_string(), "Vision API request failed: bad request");
    }

    #[test]
    fn parse_response_handles_error_payloads_without_message() {
        let response = r#"{"error":{}}"#;
        let error = parse_response(Provider::Anthropic, response).expect_err("error expected");
        assert_eq!(
            error.to_string(),
            "Vision API request failed: API returned an error response"
        );
    }

    #[test]
    fn parse_response_rejects_invalid_success_payloads() {
        let error = parse_response(Provider::Anthropic, r#"{"content":[]}"#)
            .expect_err("payload should fail");
        assert_eq!(error.to_string(), "Failed to parse vision API response");
    }

    #[test]
    fn format_output_includes_provider_attribution() {
        let output = format_output(Provider::Anthropic, "A calm orange cat.");
        assert_eq!(output, "🔍 Image Analysis (Claude):\n\nA calm orange cat.");
    }

    #[test]
    fn error_output_uses_json_contract() {
        let output = error_output(&VisionError::RequestFailed(
            "Vision API transport failed: no response from host".to_string(),
        ));
        assert_eq!(
            output,
            r#"{"error":"Vision API transport failed: no response from host"}"#
        );
    }
}
