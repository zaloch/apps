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
            let s = OsBackend::open(path)
                .with_context(|| format!("opening slide: {}", path.display()))?;
            Ok(Arc::new(s))
        }
    }
}

// ─── OpenSlide backend ───────────────────────────────────────────────────────

struct OsBackend {
    inner: openslide_rs::OpenSlide,
}

impl OsBackend {
    fn open(path: &Path) -> Result<Self> {
        let inner = openslide_rs::OpenSlide::new(path)?;
        Ok(Self { inner })
    }
}

impl Slide for OsBackend {
    fn dimensions(&self) -> Result<(u64, u64)> {
        // level 0 = full resolution
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
                // openslide-rs 2.4 uses u32 for address coordinates
                x: x.min(u32::MAX as u64) as u32,
                y: y.min(u32::MAX as u64) as u32,
            },
            level,
            size: Size { w, h },
        };
        Ok(self.inner.read_image_rgba(&region)?)
    }
}

// ─── CZI backend ─────────────────────────────────────────────────────────────
//
// czi-rs 0.1 loads subblocks on demand via the CziFile reader.
// Because CziFile contains a non-Sync BufReader we pre-decode the
// first plane into an RgbaImage and only keep that in memory.
// This is appropriate for most research/demo CZI files; very large
// multi-terabyte acquisitions would need a streaming approach.

struct CziBackend {
    width: u64,
    height: u64,
    /// Full decoded first plane, ready for cropping.
    image: RgbaImage,
}

impl CziBackend {
    fn open(path: &Path) -> Result<Self> {
        use czi_rs::{CziFile, PlaneIndex};

        let mut czi = CziFile::open(path)?;

        // Get dimensions from the bounding box of layer 0.
        let stats = czi.statistics();
        let rect = stats
            .bounding_box_layer0
            .or(stats.bounding_box)
            .with_context(|| "CZI: no bounding box found in file")?;

        let width = rect.w as u64;
        let height = rect.h as u64;

        // Read the first (default) plane.
        let plane_index = PlaneIndex::new();
        let bitmap = czi.read_plane(&plane_index)?;

        // Convert Bitmap to RgbaImage based on pixel type.
        let image = bitmap_to_rgba(&bitmap)?;

        Ok(Self { width, height, image })
    }
}

/// Convert a czi-rs Bitmap to RgbaImage, handling common pixel layouts.
fn bitmap_to_rgba(bm: &czi_rs::Bitmap) -> Result<RgbaImage> {
    use czi_rs::PixelType;

    let w = bm.width;
    let h = bm.height;
    let src = bm.as_bytes();
    let mut rgba = vec![0u8; w as usize * h as usize * 4];

    match bm.pixel_type {
        // BGR → RGB + alpha=255
        PixelType::Bgr24 => {
            for (i, chunk) in src.chunks_exact(3).enumerate() {
                let out = &mut rgba[i * 4..i * 4 + 4];
                out[0] = chunk[2]; // R
                out[1] = chunk[1]; // G
                out[2] = chunk[0]; // B
                out[3] = 255;
            }
        }
        // BGRA → RGBA
        PixelType::Bgra32 => {
            for (i, chunk) in src.chunks_exact(4).enumerate() {
                let out = &mut rgba[i * 4..i * 4 + 4];
                out[0] = chunk[2]; // R
                out[1] = chunk[1]; // G
                out[2] = chunk[0]; // B
                out[3] = chunk[3]; // A
            }
        }
        // Greyscale 8-bit → RGB grey + alpha=255
        PixelType::Gray8 => {
            for (i, &g) in src.iter().enumerate() {
                let out = &mut rgba[i * 4..i * 4 + 4];
                out[0] = g;
                out[1] = g;
                out[2] = g;
                out[3] = 255;
            }
        }
        // Greyscale 16-bit (little-endian) → scale to 8-bit
        PixelType::Gray16 => {
            for (i, chunk) in src.chunks_exact(2).enumerate() {
                let val16 = u16::from_le_bytes([chunk[0], chunk[1]]);
                let g = (val16 >> 8) as u8;
                let out = &mut rgba[i * 4..i * 4 + 4];
                out[0] = g;
                out[1] = g;
                out[2] = g;
                out[3] = 255;
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

    fn level_count(&self) -> Result<u32> {
        Ok(1)
    }

    fn level_dimensions(&self, _level: u32) -> Result<(u64, u64)> {
        Ok((self.width, self.height))
    }

    fn level_downsample(&self, _level: u32) -> Result<f64> {
        Ok(1.0)
    }

    fn best_level_for_downsample(&self, _downsample: f64) -> Result<u32> {
        Ok(0)
    }

    fn read_region(&self, x: u64, y: u64, _level: u32, w: u32, h: u32) -> Result<RgbaImage> {
        let x = x as u32;
        let y = y as u32;
        let img_w = self.width as u32;
        let img_h = self.height as u32;

        let x2 = (x + w).min(img_w);
        let y2 = (y + h).min(img_h);

        if x >= img_w || y >= img_h {
            // Return a blank tile for out-of-bounds requests.
            return Ok(RgbaImage::from_pixel(w, h, image::Rgba([0, 0, 0, 255])));
        }

        let crop = image::imageops::crop_imm(&self.image, x, y, x2 - x, y2 - y).to_image();

        // Resize to the requested size (handles edge tiles at image boundary).
        if crop.width() == w && crop.height() == h {
            Ok(crop)
        } else {
            Ok(image::imageops::resize(
                &crop,
                w,
                h,
                image::imageops::FilterType::Triangle,
            ))
        }
    }
}
