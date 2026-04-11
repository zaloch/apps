use anyhow::{bail, Context, Result};
use image::RgbaImage;
use std::{path::Path, sync::Arc};

// ─── Public trait ────────────────────────────────────────────────────────────

/// Uniform interface over every supported slide backend.
pub trait Slide: Send + Sync {
    /// Full-resolution image dimensions (width, height) in pixels.
    fn dimensions(&self) -> Result<(u64, u64)>;

    /// Number of native pyramid levels (at least 1).
    fn level_count(&self) -> Result<u32>;

    /// Dimensions at a given pyramid level.
    fn level_dimensions(&self, level: u32) -> Result<(u64, u64)>;

    /// Down-sample factor at a given pyramid level (level 0 → 1.0).
    fn level_downsample(&self, level: u32) -> Result<f64>;

    /// Best pyramid level for a requested down-sample factor.
    fn best_level_for_downsample(&self, downsample: f64) -> Result<u32>;

    /// Read an RGBA region from the slide.
    ///
    /// `x` / `y` are coordinates in **level-0** space.
    /// `w` × `h` is the region size at the chosen `level`.
    fn read_region(&self, x: u64, y: u64, level: u32, w: u32, h: u32) -> Result<RgbaImage>;
}

// ─── Factory ─────────────────────────────────────────────────────────────────

pub fn open(path: &Path) -> Result<Arc<dyn Slide>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "czi" => {
            let s = CziBackend::open(path)
                .with_context(|| format!("opening CZI: {}", path.display()))?;
            Ok(Arc::new(s))
        }
        _ => {
            #[cfg(feature = "openslide")]
            {
                let s = OsBackend::open(path)
                    .with_context(|| format!("opening slide: {}", path.display()))?;
                return Ok(Arc::new(s));
            }
            #[cfg(not(feature = "openslide"))]
            bail!(
                "Format '.{ext}' requires OpenSlide support, which is not compiled in.\n\
                 Rebuild with:  cargo build --release --features openslide\n\
                 (requires the libopenslide system library — see openslide.org)"
            );
        }
    }
}

// ─── OpenSlide backend (requires --features openslide) ───────────────────────

#[cfg(feature = "openslide")]
struct OsBackend {
    inner: openslide_rs::OpenSlide,
}

#[cfg(feature = "openslide")]
impl OsBackend {
    fn open(path: &Path) -> Result<Self> {
        let inner = openslide_rs::OpenSlide::new(path)?;
        Ok(Self { inner })
    }
}

#[cfg(feature = "openslide")]
impl Slide for OsBackend {
    fn dimensions(&self) -> Result<(u64, u64)> {
        let s = self.inner.get_level_dimensions(0)?;
        Ok((s.w as u64, s.h as u64))
    }

    fn level_count(&self) -> Result<u32> {
        Ok(self.inner.get_level_count()?)
    }

    fn level_dimensions(&self, level: u32) -> Result<(u64, u64)> {
        let s = self.inner.get_level_dimensions(level)?;
        Ok((s.w as u64, s.h as u64))
    }

    fn level_downsample(&self, level: u32) -> Result<f64> {
        Ok(self.inner.get_level_downsample(level)?)
    }

    fn best_level_for_downsample(&self, downsample: f64) -> Result<u32> {
        Ok(self.inner.get_best_level_for_downsample(downsample)?)
    }

    fn read_region(&self, x: u64, y: u64, level: u32, w: u32, h: u32) -> Result<RgbaImage> {
        use openslide_rs::{Address, Region, Size};
        let region = Region {
            address: Address {
                x: x.min(u32::MAX as u64) as u32,
                y: y.min(u32::MAX as u64) as u32,
            },
            level,
            size: Size { w, h },
        };
        Ok(self.inner.read_image_rgba(&region)?)
    }
}

// ─── CZI backend (pure Rust, no native deps) ─────────────────────────────────
//
// czi-rs 0.1 reads subblocks on demand via an internal BufReader.
// Because BufReader<File> is not Sync we pre-decode the first plane into an
// RgbaImage and keep only that in memory.  Suitable for most research/demo
// CZI files; very large multi-TB acquisitions would need a streaming approach.

struct CziBackend {
    width: u64,
    height: u64,
    image: RgbaImage,
}

impl CziBackend {
    fn open(path: &Path) -> Result<Self> {
        use czi_rs::{CziFile, PlaneIndex};

        let mut czi = CziFile::open(path)?;

        let stats = czi.statistics();
        let rect = stats
            .bounding_box_layer0
            .or(stats.bounding_box)
            .context("CZI: no bounding box found in file")?;

        let width = rect.w as u64;
        let height = rect.h as u64;

        let bitmap = czi.read_plane(&PlaneIndex::new())?;
        let image = bitmap_to_rgba(&bitmap)?;

        Ok(Self { width, height, image })
    }
}

/// Convert a czi-rs `Bitmap` to `RgbaImage`, handling common pixel layouts.
fn bitmap_to_rgba(bm: &czi_rs::Bitmap) -> Result<RgbaImage> {
    use czi_rs::PixelType;

    let w = bm.width;
    let h = bm.height;
    let src = bm.as_bytes();
    let mut rgba = vec![0u8; w as usize * h as usize * 4];

    match bm.pixel_type {
        PixelType::Bgr24 => {
            for (i, chunk) in src.chunks_exact(3).enumerate() {
                let p = &mut rgba[i * 4..i * 4 + 4];
                p[0] = chunk[2]; p[1] = chunk[1]; p[2] = chunk[0]; p[3] = 255;
            }
        }
        PixelType::Bgra32 => {
            for (i, chunk) in src.chunks_exact(4).enumerate() {
                let p = &mut rgba[i * 4..i * 4 + 4];
                p[0] = chunk[2]; p[1] = chunk[1]; p[2] = chunk[0]; p[3] = chunk[3];
            }
        }
        PixelType::Gray8 => {
            for (i, &g) in src.iter().enumerate() {
                let p = &mut rgba[i * 4..i * 4 + 4];
                p[0] = g; p[1] = g; p[2] = g; p[3] = 255;
            }
        }
        PixelType::Gray16 => {
            for (i, chunk) in src.chunks_exact(2).enumerate() {
                let g = (u16::from_le_bytes([chunk[0], chunk[1]]) >> 8) as u8;
                let p = &mut rgba[i * 4..i * 4 + 4];
                p[0] = g; p[1] = g; p[2] = g; p[3] = 255;
            }
        }
        other => bail!("CZI: unsupported pixel type {:?}", other),
    }

    RgbaImage::from_raw(w, h, rgba).context("CZI: failed to build RgbaImage")
}

impl Slide for CziBackend {
    fn dimensions(&self) -> Result<(u64, u64)> {
        Ok((self.width, self.height))
    }

    fn level_count(&self) -> Result<u32> { Ok(1) }

    fn level_dimensions(&self, _level: u32) -> Result<(u64, u64)> {
        Ok((self.width, self.height))
    }

    fn level_downsample(&self, _level: u32) -> Result<f64> { Ok(1.0) }

    fn best_level_for_downsample(&self, _downsample: f64) -> Result<u32> { Ok(0) }

    fn read_region(&self, x: u64, y: u64, _level: u32, w: u32, h: u32) -> Result<RgbaImage> {
        let (x, y) = (x as u32, y as u32);
        let (iw, ih) = (self.width as u32, self.height as u32);

        if x >= iw || y >= ih {
            return Ok(RgbaImage::from_pixel(w, h, image::Rgba([0, 0, 0, 255])));
        }

        let crop = image::imageops::crop_imm(
            &self.image, x, y,
            (x + w).min(iw) - x,
            (y + h).min(ih) - y,
        ).to_image();

        if crop.width() == w && crop.height() == h {
            Ok(crop)
        } else {
            Ok(image::imageops::resize(&crop, w, h, image::imageops::FilterType::Triangle))
        }
    }
}
