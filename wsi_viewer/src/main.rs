mod dzi;
mod routes;
mod slide;

use anyhow::Result;
use axum::{
    Router,
    routing::get,
};
use clap::Parser;
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;

#[derive(Parser)]
#[command(name = "wsi-viewer", about = "Web viewer for whole-slide images (WSI / CZI)")]
pub struct Args {
    /// Path to slide file (.svs, .ndpi, .mrxs, .czi, .tiff, …)
    #[arg(short, long)]
    pub file: PathBuf,

    /// TCP port to listen on
    #[arg(short, long, default_value = "3000")]
    pub port: u16,

    /// JPEG tile quality (1-100)
    #[arg(short = 'q', long, default_value = "80")]
    pub quality: u8,
}

pub struct AppState {
    pub slide: Arc<dyn slide::Slide>,
    pub dzi: dzi::DziInfo,
    pub quality: u8,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("wsi_viewer=info".parse()?),
        )
        .init();

    let args = Args::parse();

    tracing::info!("Opening slide: {}", args.file.display());
    let slide = slide::open(&args.file)?;

    let (width, height) = slide.dimensions()?;
    tracing::info!("Slide dimensions: {}x{}", width, height);

    let dzi = dzi::DziInfo::new(width, height, 254, 1);
    let state = Arc::new(AppState {
        slide,
        dzi,
        quality: args.quality,
    });

    let app = Router::new()
        .route("/", get(routes::index))
        .route("/dzi", get(routes::dzi_descriptor))
        // axum 0.7: path segments with dots need to be captured as a string
        .route("/dzi_files/{level}/{tile}", get(routes::dzi_tile))
        .with_state(state)
        .layer(CorsLayer::permissive());

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    tracing::info!("Listening on http://localhost:{}", args.port);
    tracing::info!("Open your browser at http://localhost:{}", args.port);

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
