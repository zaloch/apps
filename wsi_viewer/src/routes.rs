use axum::{
    Json,
    body::Body,
    extract::{Path, State},
    http::{StatusCode, header},
    response::{Html, IntoResponse, Redirect, Response},
};
use image::DynamicImage;
use serde_json::json;
use std::{io::Cursor, sync::Arc};

use crate::AppState;

// ─── UI routes ───────────────────────────────────────────────────────────────

/// Root: redirect to the viewer if there's only one slide, otherwise show the
/// file-browser so the user can pick which slide to open.
pub async fn index(State(state): State<Arc<AppState>>) -> Response {
    if state.catalogue.len() == 1 {
        Redirect::to("/viewer?id=0").into_response()
    } else {
        Html(include_str!("../static/index.html")).into_response()
    }
}

/// OpenSeadragon viewer page (reads `?id=N` from the URL at runtime via JS).
pub async fn viewer() -> Html<&'static str> {
    Html(include_str!("../static/viewer.html"))
}

// ─── API routes ──────────────────────────────────────────────────────────────

/// Return the slide catalogue as JSON.
pub async fn api_files(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let list: Vec<_> = state
        .catalogue
        .iter()
        .enumerate()
        .map(|(i, e)| json!({"id": i, "name": e.display_name}))
        .collect();
    Json(list)
}

/// Deep Zoom Image XML descriptor for slide `idx`.
pub async fn dzi_descriptor(
    Path(idx): Path<usize>,
    State(state): State<Arc<AppState>>,
) -> Response {
    match state.get_slide(idx).await {
        Ok(opened) => {
            let xml = opened.dzi.to_xml("jpeg");
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/xml; charset=utf-8")],
                xml,
            )
                .into_response()
        }
        Err(e) => {
            tracing::warn!("dzi descriptor idx={idx}: {e:#}");
            (StatusCode::NOT_FOUND, e.to_string()).into_response()
        }
    }
}

/// JPEG tile for slide `idx` at DZI level `level`, tile `{col}_{row}.jpeg`.
pub async fn dzi_tile(
    Path((idx, level, tile)): Path<(usize, u32, String)>,
    State(state): State<Arc<AppState>>,
) -> Response {
    match state.get_slide(idx).await {
        Ok(opened) => match render_tile(&opened, level, &tile, state.quality) {
            Ok(bytes) => (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "image/jpeg")],
                Body::from(bytes),
            )
                .into_response(),
            Err(e) => {
                tracing::warn!("tile idx={idx} level={level} tile={tile}: {e:#}");
                StatusCode::NOT_FOUND.into_response()
            }
        },
        Err(e) => {
            tracing::warn!("get_slide idx={idx}: {e:#}");
            StatusCode::NOT_FOUND.into_response()
        }
    }
}

// ─── Tile rendering ──────────────────────────────────────────────────────────

fn render_tile(
    opened: &crate::OpenSlide,
    dzi_level: u32,
    tile_name: &str,
    quality: u8,
) -> anyhow::Result<Vec<u8>> {
    // Parse "{col}_{row}.jpeg"
    let (col_row, _ext) = tile_name.rsplit_once('.').unwrap_or((tile_name, "jpeg"));
    let mut parts = col_row.splitn(2, '_');
    let col: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let row: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);

    let bounds = opened.dzi.tile_bounds(dzi_level, col, row);
    let dzi_ds = opened.dzi.level_downsample(dzi_level);

    let slide_level = opened.slide.best_level_for_downsample(dzi_ds)?;
    let slide_ds = opened.slide.level_downsample(slide_level)?;

    // Level-0 origin of the tile region.
    let x0 = (bounds.x0 as f64 * dzi_ds) as u64;
    let y0 = (bounds.y0 as f64 * dzi_ds) as u64;

    // Size at the chosen slide level.
    let scale = dzi_ds / slide_ds;
    let read_w = ((bounds.w as f64 * scale).ceil() as u32).max(1);
    let read_h = ((bounds.h as f64 * scale).ceil() as u32).max(1);

    let rgba = opened.slide.read_region(x0, y0, slide_level, read_w, read_h)?;

    // Resize when the slide level doesn't perfectly match this DZI level.
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

    // JPEG requires RGB (no alpha).
    let rgb = DynamicImage::ImageRgba8(rgba).into_rgb8();

    let mut buf: Vec<u8> = Vec::new();
    let mut cursor = Cursor::new(&mut buf);
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, quality);
    encoder.encode_image(&DynamicImage::ImageRgb8(rgb))?;

    Ok(buf)
}
