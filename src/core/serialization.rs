//! Format-agnostic contract input and output: JSON and XML.
//!
//! XML is treated as an alternative *syntax* for the same data model, not
//! as a second schema. Documents are transcoded to [`serde_json::Value`]
//! and then deserialized with the existing serde derives, so both formats
//! share one definition, one set of defaults and one set of validation
//! rules — and a new product supports both the moment it is added.
//!
//! (Deriving XML directly is not an option here: the data model relies on
//! `#[serde(flatten)]`, internally tagged enums and untagged enums, none of
//! which XML serde implementations support.)
//!
//! # XML conventions
//!
//! - **Elements are object fields.** `<strike_price>100</strike_price>`
//!   becomes `"strike_price": 100`.
//! - **Attributes are object fields too**, which reads naturally for the
//!   tag of a tagged enum: `<discount_curve type="flat">` is the same as
//!   `"discount_curve": { "type": "flat", ... }`.
//! - **`<item>` children make an array.** `<tenors><item>0.5</item>
//!   <item>1.0</item></tenors>` becomes `"tenors": [0.5, 1.0]`, and a
//!   single `<item>` still yields a one-element array. Repeated non-`item`
//!   siblings also collapse into an array.
//! - **Scalars are inferred**: `true`/`false` become booleans, anything
//!   parsing as a number becomes a number, an empty element becomes null,
//!   everything else stays a string (so `2027-07-17` and `C` are safe).

use std::fmt::Write as _;
use std::path::Path;

use quick_xml::events::Event;
use quick_xml::Reader;
use serde_json::{Map, Value};

/// Element name that marks array members in XML.
pub const ARRAY_ITEM: &str = "item";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Json,
    Xml,
}

impl Format {
    /// Format implied by a file extension; `None` if unrecognised.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Option<Format> {
        match path
            .as_ref()
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .as_deref()
        {
            Some("xml") => Some(Format::Xml),
            Some("json") => Some(Format::Json),
            _ => None,
        }
    }

    /// Format sniffed from document content: a leading `<` means XML.
    pub fn detect(content: &str) -> Format {
        match content.trim_start().chars().next() {
            Some('<') => Format::Xml,
            _ => Format::Json,
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            Format::Json => "json",
            Format::Xml => "xml",
        }
    }
}

// ── Input ───────────────────────────────────────────────────────────────

/// Parse a document in either format into a [`Value`].
pub fn parse_value(content: &str, format: Format) -> Result<Value, String> {
    match format {
        Format::Json => serde_json::from_str(content).map_err(|e| format!("invalid JSON: {e}")),
        Format::Xml => xml_to_value(content),
    }
}

/// Parse a document into any deserializable type, in either format.
pub fn parse<T: serde::de::DeserializeOwned>(content: &str, format: Format) -> Result<T, String> {
    let value = parse_value(content, format)?;
    serde_json::from_value(value).map_err(|e| format!("document does not match the schema: {e}"))
}

/// Transcode an XML document into the equivalent [`Value`].
pub fn xml_to_value(xml: &str) -> Result<Value, String> {
    let mut reader = Reader::from_str(xml);
    // text is accumulated raw and trimmed when the element closes, so
    // indentation is discarded without collapsing interior whitespace
    reader.config_mut().expand_empty_elements = false;

    // stack of partially built elements
    let mut stack: Vec<Node> = Vec::new();
    let mut root: Option<(String, Value)> = None;

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => stack.push(Node::start(&e)?),
            Ok(Event::Empty(e)) => {
                let node = Node::start(&e)?;
                let (name, value) = node.finish();
                attach(&mut stack, &mut root, name, value)?;
            }
            Ok(Event::Text(e)) => {
                if let Some(node) = stack.last_mut() {
                    let text = e
                        .decode()
                        .map_err(|err| format!("invalid text content: {err}"))?;
                    node.text.push_str(text.as_ref());
                }
            }
            Ok(Event::CData(e)) => {
                if let Some(node) = stack.last_mut() {
                    let text = String::from_utf8(e.into_inner().into_owned())
                        .map_err(|err| format!("invalid CDATA: {err}"))?;
                    node.text.push_str(&text);
                }
            }
            // entity references arrive as their own events
            Ok(Event::GeneralRef(e)) => {
                if let Some(node) = stack.last_mut() {
                    node.text.push_str(&resolve_entity(&e)?);
                }
            }
            Ok(Event::End(_)) => {
                let node = stack.pop().ok_or_else(|| "unbalanced closing tag".to_string())?;
                let (name, value) = node.finish();
                attach(&mut stack, &mut root, name, value)?;
            }
            Ok(Event::Eof) => break,
            Ok(_) => {} // declaration, comments, processing instructions
            Err(e) => return Err(format!("malformed XML at byte {}: {e}", reader.buffer_position())),
        }
    }
    if !stack.is_empty() {
        return Err("unbalanced XML: unclosed elements".to_string());
    }
    match root {
        // the document element is the top-level object
        Some((_, value)) => Ok(value),
        None => Err("empty XML document".to_string()),
    }
}

struct Node {
    name: String,
    /// attributes, in document order
    attrs: Vec<(String, Value)>,
    /// child elements, in document order
    children: Vec<(String, Value)>,
    text: String,
}

impl Node {
    fn start(e: &quick_xml::events::BytesStart) -> Result<Node, String> {
        let name = String::from_utf8(e.name().as_ref().to_vec())
            .map_err(|err| format!("invalid element name: {err}"))?;
        let mut attrs = Vec::new();
        for attr in e.attributes() {
            let attr = attr.map_err(|err| format!("invalid attribute in <{name}>: {err}"))?;
            let key = String::from_utf8(attr.key.as_ref().to_vec())
                .map_err(|err| format!("invalid attribute name: {err}"))?;
            let raw = attr
                .unescape_value()
                .map_err(|err| format!("invalid attribute value in <{name}>: {err}"))?;
            attrs.push((key, infer_scalar(raw.as_ref())));
        }
        Ok(Node { name, attrs, children: Vec::new(), text: String::new() })
    }

    fn finish(self) -> (String, Value) {
        let Node { name, attrs, children, text } = self;

        // an element whose children are all <item> is an array
        if !children.is_empty() && children.iter().all(|(n, _)| n == ARRAY_ITEM) {
            let items = children.into_iter().map(|(_, v)| v).collect();
            return (name, Value::Array(items));
        }

        if children.is_empty() && attrs.is_empty() {
            let trimmed = text.trim();
            return (name, infer_scalar(trimmed));
        }

        let mut map = Map::new();
        for (key, value) in attrs {
            map.insert(key, value);
        }
        // repeated sibling names collapse into an array
        for (key, value) in children {
            match map.get_mut(&key) {
                Some(Value::Array(existing)) => existing.push(value),
                Some(slot) => {
                    let previous = slot.take();
                    *slot = Value::Array(vec![previous, value]);
                }
                None => {
                    map.insert(key, value);
                }
            }
        }
        (name, Value::Object(map))
    }
}

/// Resolve an entity reference: the five predefined XML entities plus
/// numeric character references (`&#38;`, `&#x26;`).
fn resolve_entity(e: &quick_xml::events::BytesRef) -> Result<String, String> {
    if e.is_char_ref() {
        return match e.resolve_char_ref() {
            Ok(Some(c)) => Ok(c.to_string()),
            Ok(None) => Err("unresolvable character reference".to_string()),
            Err(err) => Err(format!("invalid character reference: {err}")),
        };
    }
    let name = e.decode().map_err(|err| format!("invalid entity reference: {err}"))?;
    match name.as_ref() {
        "amp" => Ok("&".to_string()),
        "lt" => Ok("<".to_string()),
        "gt" => Ok(">".to_string()),
        "quot" => Ok("\"".to_string()),
        "apos" => Ok("'".to_string()),
        other => Err(format!(
            "unknown entity '&{other};' (only the predefined XML entities are supported)"
        )),
    }
}

fn attach(
    stack: &mut [Node],
    root: &mut Option<(String, Value)>,
    name: String,
    value: Value,
) -> Result<(), String> {
    match stack.last_mut() {
        Some(parent) => {
            parent.children.push((name, value));
            Ok(())
        }
        None => {
            if root.is_some() {
                return Err("XML documents must have a single root element".to_string());
            }
            *root = Some((name, value));
            Ok(())
        }
    }
}

/// Infer a JSON scalar from XML text content.
fn infer_scalar(text: &str) -> Value {
    let t = text.trim();
    if t.is_empty() {
        return Value::Null;
    }
    match t {
        "true" => return Value::Bool(true),
        "false" => return Value::Bool(false),
        "null" => return Value::Null,
        _ => {}
    }
    // only treat as a number when the whole token is numeric, so dates and
    // codes ("2027-07-17", "C") stay strings
    if let Ok(i) = t.parse::<i64>() {
        return Value::Number(i.into());
    }
    if let Ok(f) = t.parse::<f64>() {
        if f.is_finite() {
            if let Some(n) = serde_json::Number::from_f64(f) {
                return Value::Number(n);
            }
        }
    }
    Value::String(t.to_string())
}

// ── Output ──────────────────────────────────────────────────────────────

/// Render a list of contract results in the requested format.
///
/// JSON output is a well-formed array; XML output wraps the results in a
/// `<results>` document element.
pub fn render_results(results: &[Value], format: Format) -> String {
    let mut array = Value::Array(results.to_vec());
    strip_nulls(&mut array);
    match format {
        Format::Json => serde_json::to_string_pretty(&array).unwrap_or_else(|_| "[]".to_string()),
        Format::Xml => value_to_xml(&array, "results"),
    }
}

/// Render a single value in the requested format.
pub fn render_value(value: &Value, format: Format, root: &str) -> String {
    let mut value = value.clone();
    strip_nulls(&mut value);
    match format {
        Format::Json => serde_json::to_string_pretty(&value).unwrap_or_default(),
        Format::Xml => value_to_xml(&value, root),
    }
}

/// Drop null object fields recursively.
///
/// Every nullable field in the data model is an `Option`, for which an
/// absent key and an explicit null are equivalent on the way back in, so
/// this keeps output readable without changing what it means.
pub fn strip_nulls(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.retain(|_, v| !v.is_null());
            for v in map.values_mut() {
                strip_nulls(v);
            }
        }
        Value::Array(items) => {
            for item in items {
                strip_nulls(item);
            }
        }
        _ => {}
    }
}

/// Serialize a [`Value`] as an XML document with `root` as the document
/// element. Arrays are written as `<item>` children, mirroring the input
/// convention, so output can be fed back in as input.
pub fn value_to_xml(value: &Value, root: &str) -> String {
    let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    write_element(&mut out, root, value, 0);
    out
}

fn write_element(out: &mut String, name: &str, value: &Value, depth: usize) {
    let pad = "  ".repeat(depth);
    match value {
        Value::Null => {
            let _ = writeln!(out, "{pad}<{name}/>");
        }
        Value::Bool(_) | Value::Number(_) | Value::String(_) => {
            let text = match value {
                Value::String(s) => escape_text(s),
                other => other.to_string(),
            };
            let _ = writeln!(out, "{pad}<{name}>{text}</{name}>");
        }
        Value::Array(items) => {
            if items.is_empty() {
                let _ = writeln!(out, "{pad}<{name}/>");
                return;
            }
            let _ = writeln!(out, "{pad}<{name}>");
            for item in items {
                write_element(out, ARRAY_ITEM, item, depth + 1);
            }
            let _ = writeln!(out, "{pad}</{name}>");
        }
        Value::Object(map) => {
            if map.is_empty() {
                let _ = writeln!(out, "{pad}<{name}/>");
                return;
            }
            let _ = writeln!(out, "{pad}<{name}>");
            for (key, child) in map {
                write_element(out, key, child, depth + 1);
            }
            let _ = writeln!(out, "{pad}</{name}>");
        }
    }
}

fn escape_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn detects_format_from_content_and_path() {
        assert_eq!(Format::detect("  { \"a\": 1 }"), Format::Json);
        assert_eq!(Format::detect("\n<?xml version=\"1.0\"?><a/>"), Format::Xml);
        assert_eq!(Format::detect("<contracts/>"), Format::Xml);
        assert_eq!(Format::from_path("in.xml"), Some(Format::Xml));
        assert_eq!(Format::from_path("in.JSON"), Some(Format::Json));
        assert_eq!(Format::from_path("in.txt"), None);
    }

    #[test]
    fn scalars_are_inferred_without_eating_dates_or_codes() {
        assert_eq!(infer_scalar("100"), json!(100));
        assert_eq!(infer_scalar(" 0.30 "), json!(0.30));
        assert_eq!(infer_scalar("-1.5e-3"), json!(-0.0015));
        assert_eq!(infer_scalar("true"), json!(true));
        assert_eq!(infer_scalar(""), Value::Null);
        // must stay strings
        assert_eq!(infer_scalar("2027-07-17"), json!("2027-07-17"));
        assert_eq!(infer_scalar("C"), json!("C"));
        assert_eq!(infer_scalar("down_out"), json!("down_out"));
        assert_eq!(infer_scalar("Act365"), json!("Act365"));
    }

    #[test]
    fn elements_and_attributes_both_become_fields() {
        let value = xml_to_value(
            r#"<curve type="flat"><rate>0.05</rate><day_count>Act365</day_count></curve>"#,
        )
        .unwrap();
        assert_eq!(value, json!({"type": "flat", "rate": 0.05, "day_count": "Act365"}));
    }

    #[test]
    fn item_children_make_arrays_including_single_element() {
        let value = xml_to_value("<tenors><item>0.5</item><item>1.0</item></tenors>").unwrap();
        assert_eq!(value, json!([0.5, 1.0]));
        let single = xml_to_value("<tenors><item>0.5</item></tenors>").unwrap();
        assert_eq!(single, json!([0.5]));
    }

    #[test]
    fn nested_arrays_round_trip() {
        let xml = "<vols><item><item>0.32</item><item>0.30</item></item>\
                   <item><item>0.33</item><item>0.31</item></item></vols>";
        assert_eq!(xml_to_value(xml).unwrap(), json!([[0.32, 0.30], [0.33, 0.31]]));
    }

    #[test]
    fn repeated_siblings_collapse_into_an_array() {
        let value = xml_to_value("<root><tag>a</tag><tag>b</tag><other>c</other></root>").unwrap();
        assert_eq!(value, json!({"tag": ["a", "b"], "other": "c"}));
    }

    #[test]
    fn empty_and_self_closing_elements_are_null() {
        let value = xml_to_value("<root><a/><b></b><c>1</c></root>").unwrap();
        assert_eq!(value, json!({"a": null, "b": null, "c": 1}));
    }

    #[test]
    fn entities_and_cdata_are_decoded() {
        let value = xml_to_value("<root><a>A &amp; B</a><b><![CDATA[x < y]]></b></root>").unwrap();
        assert_eq!(value, json!({"a": "A & B", "b": "x < y"}));
    }

    #[test]
    fn declaration_and_comments_are_ignored() {
        let value =
            xml_to_value("<?xml version=\"1.0\"?><!-- note --><root><a>1</a></root>").unwrap();
        assert_eq!(value, json!({"a": 1}));
    }

    #[test]
    fn malformed_documents_are_reported() {
        assert!(xml_to_value("<root><a></root>").is_err());
        assert!(xml_to_value("").is_err());
        assert!(xml_to_value("not xml at all").is_err());
    }

    #[test]
    fn value_to_xml_round_trips_through_the_reader() {
        let original = json!({
            "asset": "EQ",
            "contracts": [
                {"action": "PV", "strike_price": 100.0, "flag": true, "missing": null},
                {"action": "PV", "tenors": [0.5, "2028-07-16"], "nested": [[1.0, 2.0]]}
            ]
        });
        let xml = value_to_xml(&original, "root");
        let back = xml_to_value(&xml).unwrap();
        assert_eq!(back, original, "\nXML was:\n{xml}");
    }

    #[test]
    fn xml_special_characters_survive_a_round_trip() {
        let original = json!({"name": "Smith & Co <\"AAA\">"});
        let back = xml_to_value(&value_to_xml(&original, "root")).unwrap();
        assert_eq!(back, original);
    }

    #[test]
    fn null_fields_are_dropped_from_output() {
        let mut v = json!({"a": 1, "b": null, "c": {"d": null, "e": 2}, "f": [{"g": null}]});
        strip_nulls(&mut v);
        assert_eq!(v, json!({"a": 1, "c": {"e": 2}, "f": [{}]}));
    }

    #[test]
    fn rendered_output_is_valid_in_both_formats() {
        let results = vec![json!({"contract": {"action": "PV", "skip": null}, "output": {"pv": 1.5}})];
        let as_json: Value = serde_json::from_str(&render_results(&results, Format::Json)).unwrap();
        let as_xml = xml_to_value(&render_results(&results, Format::Xml)).unwrap();
        assert_eq!(as_json, as_xml, "both formats must carry the same data");
        assert_eq!(as_json[0]["output"]["pv"], json!(1.5));
        assert!(as_json[0]["contract"].get("skip").is_none());
    }

    #[test]
    fn parse_dispatches_on_format() {
        #[derive(serde::Deserialize, PartialEq, Debug)]
        struct Doc {
            a: i32,
            b: String,
        }
        let from_json: Doc = parse(r#"{"a": 1, "b": "x"}"#, Format::Json).unwrap();
        let from_xml: Doc = parse("<doc><a>1</a><b>x</b></doc>", Format::Xml).unwrap();
        assert_eq!(from_json, from_xml);
    }
}
