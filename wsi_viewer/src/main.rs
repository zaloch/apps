mod dzi;
mod routes;
mod slide;

use anyhow::{bail, Result};
use axum::{Router, routing::get};
use clap::Parser;
use std::{
    collections::HashMap,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;

// ─── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "wsi-viewer", about = "Web viewer for whole-slide images (WSI / CZI)")]
pub struct Args {
    /// Path to a slide file OR a directory containing slides.
    /// Supported formats: .svs .ndpi .mrxs .scn .czi .tif .tiff .bif .vms .vmu
    #[arg(short = 'f', long)]
    pub file: Option<PathBuf>,

    #[arg(short = 'd', long)]
    pub dir: Option<PathBuf>,

    /// TCP port to listen on
    #[arg(short, long, default_value = "3000")]
    pub port: u16,

    /// JPEG tile quality (1-100)
    #[arg(short = 'q', long, default_value = "80")]
    pub quality: u8,
}

// ─── Application state ───────────────────────────────────────────────────────

/// One entry in the slide catalogue.
#[derive(Clone)]
pub struct SlideEntry {
    pub display_name: String,
    pub path: PathBuf,
}

/// Per-slide data opened on demand.
#[derive(Clone)]
pub struct OpenSlide {
    pub slide: Arc<dyn slide::Slide>,
    pub dzi: dzi::DziInfo,
}

pub struct AppState {
    pub catalogue: Vec<SlideEntry>,
    pub quality: u8,
    /// Lazy cache: index → open slide data.
    pub cache: RwLock<HashMap<usize, OpenSlide>>,
}

impl AppState {
    /// Return (possibly cached) opened slide for the given catalogue index.
    pub async fn get_slide(&self, idx: usize) -> Result<OpenSlide> {
        // Fast path: already open.
        {
            let guard = self.cache.read().await;
            if let Some(entry) = guard.get(&idx) {
                return Ok(entry.clone());
            }
        }

        // Slow path: open from disk (on a blocking thread so we don't stall the runtime).
        let path = self
            .catalogue
            .get(idx)
            .map(|e| e.path.clone())
            .ok_or_else(|| anyhow::anyhow!("slide index {} out of range", idx))?;

        let slide = tokio::task::spawn_blocking(move || slide::open(&path)).await??;
        let (w, h) = slide.dimensions()?;
        let dzi = dzi::DziInfo::new(w, h, 254, 1);

        let entry = OpenSlide { slide, dzi };
        {
            let mut guard = self.cache.write().await;
            guard.entry(idx).or_insert_with(|| entry.clone());
        }
        Ok(entry)
    }
}

// ─── Known WSI / CZI extensions ──────────────────────────────────────────────

const WSI_EXTS: &[&str] = &[
    "svs", "ndpi", "mrxs", "scn", "czi",
    "tif", "tiff", "bif", "vms", "vmu", "svslide",
];

fn is_slide(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| WSI_EXTS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn scan_dir(dir: &Path) -> Result<Vec<SlideEntry>> {
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && is_slide(&path) {
            let display_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();
            entries.push(SlideEntry { display_name, path });
        }
    }
    entries.sort_by(|a, b| a.display_name.cmp(&b.display_name));
    Ok(entries)
}

// ─── Entry point ─────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("wsi_viewer=info".parse()?),
        )
        .init();

    let args = Args::parse();

    // Determine the catalogue from --file / --dir (or a bare positional path).
    let catalogue: Vec<SlideEntry> = match (&args.file, &args.dir) {
        (Some(f), None) => {
            if f.is_dir() {
                tracing::info!("Scanning directory: {}", f.display());
                scan_dir(f)?
            } else {
                vec![SlideEntry {
                    display_name: f
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("slide")
                        .to_string(),
                    path: f.clone(),
                }]
            }
        }
        (None, Some(d)) => {
            tracing::info!("Scanning directory: {}", d.display());
            scan_dir(d)?
        }
        (Some(_), Some(_)) => bail!("Specify either --file or --dir, not both"),
        (None, None) => bail!("Provide --file <slide> or --dir <directory>"),
    };

    if catalogue.is_empty() {
        bail!("No slide files found. Supported extensions: {}", WSI_EXTS.join(", "));
    }

    tracing::info!("Catalogue: {} slide(s)", catalogue.len());
    for (i, e) in catalogue.iter().enumerate() {
        tracing::info!("  [{i}] {}", e.display_name);
    }

    let state = Arc::new(AppState {
        catalogue,
        quality: args.quality,
        cache: RwLock::new(HashMap::new()),
    });

    let app = Router::new()
        // UI
        .route("/", get(routes::index))
        .route("/viewer", get(routes::viewer))
        // API
        .route("/api/files", get(routes::api_files))
        .route("/slide/{idx}/dzi", get(routes::dzi_descriptor))
        // OpenSeadragon DZI convention: given /slide/N/dzi as descriptor URL,
        // it requests tiles from /slide/N/dzi_files/{level}/{col}_{row}.jpeg
        .route("/slide/{idx}/dzi_files/{level}/{tile}", get(routes::dzi_tile))
        .with_state(state)
        .layer(CorsLayer::permissive());

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    tracing::info!("Listening on http://localhost:{}", args.port);

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
