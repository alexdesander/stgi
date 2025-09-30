//! Declaration of the atlas.json files (from msdf-atlas-gen) so we can just parse it using serde_json.

use serde::Deserialize;

/// The top-level JSON document ------------------------------------------------
#[derive(Debug, Deserialize)]
pub struct GlyphAtlasData {
    pub atlas: AtlasHeader,
    pub metrics: Metrics,
    pub glyphs: Vec<Glyph>,
    pub kerning: Vec<KerningPair>,
}

/// ---------------------------------------------------------------------------
#[derive(Debug, Deserialize)]
pub struct AtlasHeader {
    #[serde(rename = "type")]
    pub atlas_type: String, // "msdf"
    #[serde(rename = "distanceRange")]
    pub distance_range: f32,
    #[serde(rename = "distanceRangeMiddle")]
    pub distance_range_middle: f32,
    pub size: u32,
    pub width: u32,
    pub height: u32,
    #[serde(rename = "yOrigin")]
    pub y_origin: String, // "bottom"
}

/// ---------------------------------------------------------------------------
#[derive(Debug, Deserialize)]
pub struct Metrics {
    #[serde(rename = "emSize")]
    pub em_size: f32,
    #[serde(rename = "lineHeight")]
    pub line_height: f32,
    pub ascender: f32,
    pub descender: f32,
    #[serde(rename = "underlineY")]
    pub underline_y: f32,
    #[serde(rename = "underlineThickness")]
    pub underline_thickness: f32,
}

/// ---------------------------------------------------------------------------
#[derive(Debug, Deserialize)]
pub struct Glyph {
    pub unicode: u32,
    pub advance: f32,

    #[serde(rename = "planeBounds", skip_serializing_if = "Option::is_none")]
    pub plane_bounds: Option<Bounds>,

    #[serde(rename = "atlasBounds", skip_serializing_if = "Option::is_none")]
    pub atlas_bounds: Option<Bounds>,
}

/// JSON uses the same layout for plane and atlas bounds -----------------------
#[derive(Debug, Deserialize)]
pub struct Bounds {
    pub left: f32,
    pub bottom: f32,
    pub right: f32,
    pub top: f32,
}

/// ---------------------------------------------------------------------------
#[derive(Debug, Deserialize)]
pub struct KerningPair {
    #[serde(rename = "unicode1")]
    pub first: u32,
    #[serde(rename = "unicode2")]
    pub second: u32,
    pub advance: f32,
}
