use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Response},
};
use image::DynamicImage;
use std::{io::Cursor, sync::Arc};

use crate::AppState;

// ─── Index ───────────────────────────────────────────────────────────────────

pub async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

// ─── DZI descriptor ──────────────────────────────────────────────────────────

pub async fn dzi_descriptor(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let xml = state.dzi.to_xml("jpeg");
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/xml; charset=utf-8")],
        xml,
    )
}

// ─── Tile endpoint ───────────────────────────────────────────────────────────

/// `tile` has the form `{col}_{row}.jpeg` (or .png).
pub async fn dzi_tile(
    Path((level, tile)): Path<(u32, String)>,
    State(state): State<Arc<AppState>>,
) -> Response {
    match render_tile(&state, level, &tile) {
        Ok(bytes) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "image/jpeg")],
            Body::from(bytes),
        )
            .into_response(),
        Err(e) => {
            tracing::warn!("tile error level={level} tile={tile}: {e:#}");
            StatusCode::NOT_FOUND.into_response()
        }
    }
}

// ─── Tile rendering ──────────────────────────────────────────────────────────

fn render_tile(state: &AppState, dzi_level: u32, tile_name: &str) -> anyhow::Result<Vec<u8>> {
    // Parse "{col}_{row}.jpeg"
    let (col_row, _ext) = tile_name
        .rsplit_once('.')
        .unwrap_or((tile_name, "jpeg"));
    let mut parts = col_row.splitn(2, '_');
    let col: u32 = parts
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let row: u32 = parts
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // DZI geometry for this tile.
    let bounds = state.dzi.tile_bounds(dzi_level, col, row);
    let dzi_downsample = state.dzi.level_downsample(dzi_level);

    // Choose the best native slide level.
    let slide_level = state.slide.best_level_for_downsample(dzi_downsample)?;
    let slide_ds = state.slide.level_downsample(slide_level)?;

    // Map the DZI canvas position back to level-0 coordinates.
    let x0 = (bounds.x0 as f64 * dzi_downsample) as u64;
    let y0 = (bounds.y0 as f64 * dzi_downsample) as u64;

    // Size of the region at the chosen slide level.
    let scale_factor = dzi_downsample / slide_ds; // ≥ 1.0
    let read_w = (bounds.w as f64 * scale_factor).ceil() as u32;
    let read_h = (bounds.h as f64 * scale_factor).ceil() as u32;

    let rgba = state.slide.read_region(x0, y0, slide_level, read_w, read_h)?;

    // Resize to the exact tile dimensions when the slide level isn't a
    // perfect match for this DZI level.
    let rgba = if read_w != bounds.w || read_h != bounds.h {
        image::imageops::resize(
            &rgba,
            bounds.w,
            bounds.h,
            image::imageops::FilterType::Lanczos3,
        )
    } else {
        rgba
    };

    // JPEG does not support alpha; convert to RGB.
    let rgb = DynamicImage::ImageRgba8(rgba).into_rgb8();

    // Encode.
    let mut buf: Vec<u8> = Vec::new();
    let mut cursor = Cursor::new(&mut buf);
    let mut encoder =
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, state.quality);
    encoder.encode_image(&DynamicImage::ImageRgb8(rgb))?;

    Ok(buf)
}
