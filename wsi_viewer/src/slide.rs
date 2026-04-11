use anyhow::{bail, Context, Result};
use image::RgbaImage;
use std::{
    path::Path,
    sync::{Arc, Mutex},
};

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
    /// `x` / `y` are level-0 coordinates; `w × h` is the requested size at `level`.
    fn read_region(&self, x: u64, y: u64, level: u32, w: u32, h: u32) -> Result<RgbaImage>;
}

// ─── Factory — try each backend in priority order ────────────────────────────

pub fn open(path: &Path) -> Result<Arc<dyn Slide>> {
    let name = path.display();

    // 1. bioformats — broadest format coverage
    match BioBackend::open(path) {
        Ok(s) => {
            tracing::debug!("bioformats opened {name}");
            return Ok(Arc::new(s));
        }
        Err(e) => tracing::debug!("bioformats failed for {name}: {e:#}"),
    }

    // 2. czi-rs — dedicated Zeiss CZI reader
    match CziBackend::open(path) {
        Ok(s) => {
            tracing::debug!("czi-rs opened {name}");
            return Ok(Arc::new(s));
        }
        Err(e) => tracing::debug!("czi-rs failed for {name}: {e:#}"),
    }

    // 3. openslide-pure-rs — pure-Rust Mirax + future formats
    match OsBackend::open(path) {
        Ok(s) => {
            tracing::debug!("openslide-pure-rs opened {name}");
            return Ok(Arc::new(s));
        }
        Err(e) => tracing::debug!("openslide-pure-rs failed for {name}: {e:#}"),
    }

    bail!("No backend could open '{name}'. Supported: SVS, NDPI, SCN, CZI, OME-TIFF, MRXS, TIFF, and more.")
}

// ─── 1. bioformats backend ───────────────────────────────────────────────────
//
// FormatReader is Send+Sync by trait bound, so Mutex<ImageReader> is fine.

struct BioBackend {
    /// Mutable reader wrapped for concurrent tile requests.
    reader: Mutex<bioformats::ImageReader>,
    /// Per-level (width, height), index 0 = full resolution.
    level_dims: Vec<(u64, u64)>,
    /// Per-level down-sample factor, index 0 = 1.0.
    level_downsamples: Vec<f64>,
    // Pixel format constants cached at open time.
    pixel_type: bioformats::PixelType,
    is_rgb: bool,
    is_interleaved: bool,
    size_c: u32,
}

impl BioBackend {
    fn open(path: &Path) -> Result<Self> {
        let mut reader = bioformats::ImageReader::open(path)?;
        let n = reader.resolution_count().max(1);

        // Cache per-level dimensions.
        let mut level_dims: Vec<(u64, u64)> = Vec::with_capacity(n);
        for lvl in 0..n {
            reader.set_resolution(lvl).ok();
            let m = reader.metadata();
            level_dims.push((m.size_x as u64, m.size_y as u64));
        }
        reader.set_resolution(0).ok();

        // Sanity-check: width must be > 0.
        if level_dims[0].0 == 0 {
            bail!("bioformats: reported zero width for '{}'", path.display());
        }

        let base_w = level_dims[0].0 as f64;
        let level_downsamples: Vec<f64> = level_dims
            .iter()
            .map(|&(w, _)| if w == 0 { f64::MAX } else { base_w / w as f64 })
            .collect();

        // Cache pixel format fields before moving `reader` into the Mutex.
        let pixel_type   = reader.metadata().pixel_type;
        let is_rgb       = reader.metadata().is_rgb;
        let is_interleaved = reader.metadata().is_interleaved;
        let size_c       = reader.metadata().size_c;

        Ok(Self {
            reader: Mutex::new(reader),
            level_dims,
            level_downsamples,
            pixel_type,
            is_rgb,
            is_interleaved,
            size_c,
        })
    }
}

impl Slide for BioBackend {
    fn dimensions(&self) -> Result<(u64, u64)> {
        Ok(self.level_dims[0])
    }

    fn level_count(&self) -> Result<u32> {
        Ok(self.level_dims.len() as u32)
    }

    fn level_dimensions(&self, level: u32) -> Result<(u64, u64)> {
        self.level_dims
            .get(level as usize)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("level {level} out of range"))
    }

    fn level_downsample(&self, level: u32) -> Result<f64> {
        self.level_downsamples
            .get(level as usize)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("level {level} out of range"))
    }

    fn best_level_for_downsample(&self, downsample: f64) -> Result<u32> {
        let mut best = 0u32;
        for (i, &ds) in self.level_downsamples.iter().enumerate() {
            if ds <= downsample {
                best = i as u32;
            }
        }
        Ok(best)
    }

    fn read_region(&self, x: u64, y: u64, level: u32, w: u32, h: u32) -> Result<RgbaImage> {
        let ds = self.level_downsamples.get(level as usize).copied().unwrap_or(1.0);
        // Convert level-0 coordinates to this level's coordinate space.
        let lx = (x as f64 / ds) as u32;
        let ly = (y as f64 / ds) as u32;

        let mut reader = self.reader.lock().unwrap();
        reader.set_resolution(level as usize).ok();
        let bytes = reader.open_bytes_region(0, lx, ly, w, h)?;

        bytes_to_rgba(&bytes, w, h, self.pixel_type, self.is_rgb, self.is_interleaved, self.size_c)
    }
}

/// Convert raw `open_bytes_region` bytes to an RGBA image.
fn bytes_to_rgba(
    bytes: &[u8],
    w: u32,
    h: u32,
    pixel_type: bioformats::PixelType,
    is_rgb: bool,
    is_interleaved: bool,
    size_c: u32,
) -> Result<RgbaImage> {
    use bioformats::PixelType;

    let n = w as usize * h as usize;
    let mut rgba = vec![255u8; n * 4]; // alpha defaults to opaque

    match (pixel_type, is_rgb, is_interleaved, size_c) {
        // 8-bit RGB interleaved (most common brightfield WSI)
        (PixelType::Uint8, true, true, 3) => {
            for (i, chunk) in bytes.chunks_exact(3).take(n).enumerate() {
                rgba[i * 4]     = chunk[0];
                rgba[i * 4 + 1] = chunk[1];
                rgba[i * 4 + 2] = chunk[2];
            }
        }
        // 8-bit RGBA interleaved
        (PixelType::Uint8, true, true, 4) => {
            for (i, chunk) in bytes.chunks_exact(4).take(n).enumerate() {
                rgba[i * 4..i * 4 + 4].copy_from_slice(chunk);
            }
        }
        // 8-bit RGB planar (RRRGGGBBB)
        (PixelType::Uint8, true, false, 3) if bytes.len() >= 3 * n => {
            for i in 0..n {
                rgba[i * 4]     = bytes[i];
                rgba[i * 4 + 1] = bytes[n + i];
                rgba[i * 4 + 2] = bytes[2 * n + i];
            }
        }
        // 8-bit greyscale
        (PixelType::Uint8, false, _, 1) => {
            for (i, &g) in bytes.iter().take(n).enumerate() {
                rgba[i * 4]     = g;
                rgba[i * 4 + 1] = g;
                rgba[i * 4 + 2] = g;
            }
        }
        // 16-bit greyscale → scale to 8-bit
        (PixelType::Uint16, false, _, 1) => {
            for (i, chunk) in bytes.chunks_exact(2).take(n).enumerate() {
                let g = (u16::from_le_bytes([chunk[0], chunk[1]]) >> 8) as u8;
                rgba[i * 4]     = g;
                rgba[i * 4 + 1] = g;
                rgba[i * 4 + 2] = g;
            }
        }
        // 16-bit RGB interleaved → scale to 8-bit
        (PixelType::Uint16, true, true, 3) => {
            for (i, chunk) in bytes.chunks_exact(6).take(n).enumerate() {
                rgba[i * 4]     = (u16::from_le_bytes([chunk[0], chunk[1]]) >> 8) as u8;
                rgba[i * 4 + 1] = (u16::from_le_bytes([chunk[2], chunk[3]]) >> 8) as u8;
                rgba[i * 4 + 2] = (u16::from_le_bytes([chunk[4], chunk[5]]) >> 8) as u8;
            }
        }
        // 16-bit RGB planar → scale to 8-bit
        (PixelType::Uint16, true, false, 3) if bytes.len() >= 6 * n => {
            let p = n * 2;
            for i in 0..n {
                rgba[i * 4]     = (u16::from_le_bytes([bytes[i*2], bytes[i*2+1]]) >> 8) as u8;
                rgba[i * 4 + 1] = (u16::from_le_bytes([bytes[p + i*2], bytes[p + i*2+1]]) >> 8) as u8;
                rgba[i * 4 + 2] = (u16::from_le_bytes([bytes[2*p + i*2], bytes[2*p + i*2+1]]) >> 8) as u8;
            }
        }
        // Generic fallback: treat first 1-4 bytes per pixel as RGBA components.
        _ => {
            let bpp = (bytes.len() / n).clamp(1, 4);
            for i in 0..n.min(bytes.len() / bpp) {
                let base = i * bpp;
                rgba[i * 4]     = bytes[base];
                rgba[i * 4 + 1] = bytes.get(base + 1).copied().unwrap_or(bytes[base]);
                rgba[i * 4 + 2] = bytes.get(base + 2).copied().unwrap_or(bytes[base]);
                if bpp == 4 { rgba[i * 4 + 3] = bytes[base + 3]; }
            }
        }
    }

    RgbaImage::from_raw(w, h, rgba)
        .context("bioformats: failed to build RgbaImage from pixel bytes")
}

// ─── 2. czi-rs backend ───────────────────────────────────────────────────────
//
// czi-rs 0.1 reads subblocks via an internal BufReader (not Sync).
// We decode the first plane once into an RgbaImage at open time and
// crop/resize from that in-memory buffer for every tile request.

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
            .context("czi-rs: no bounding box in file")?;

        let width  = rect.w as u64;
        let height = rect.h as u64;
        let bitmap = czi.read_plane(&PlaneIndex::new())?;
        let image  = czi_bitmap_to_rgba(&bitmap)?;

        Ok(Self { width, height, image })
    }
}

fn czi_bitmap_to_rgba(bm: &czi_rs::Bitmap) -> Result<RgbaImage> {
    use czi_rs::PixelType;

    let (w, h) = (bm.width, bm.height);
    let src    = bm.as_bytes();
    let mut rgba = vec![0u8; w as usize * h as usize * 4];

    match bm.pixel_type {
        PixelType::Bgr24  => {
            for (i, c) in src.chunks_exact(3).enumerate() {
                let p = &mut rgba[i * 4..i * 4 + 4];
                p[0] = c[2]; p[1] = c[1]; p[2] = c[0]; p[3] = 255;
            }
        }
        PixelType::Bgra32 => {
            for (i, c) in src.chunks_exact(4).enumerate() {
                let p = &mut rgba[i * 4..i * 4 + 4];
                p[0] = c[2]; p[1] = c[1]; p[2] = c[0]; p[3] = c[3];
            }
        }
        PixelType::Gray8  => {
            for (i, &g) in src.iter().enumerate() {
                let p = &mut rgba[i * 4..i * 4 + 4];
                p[0] = g; p[1] = g; p[2] = g; p[3] = 255;
            }
        }
        PixelType::Gray16 => {
            for (i, c) in src.chunks_exact(2).enumerate() {
                let g = (u16::from_le_bytes([c[0], c[1]]) >> 8) as u8;
                let p = &mut rgba[i * 4..i * 4 + 4];
                p[0] = g; p[1] = g; p[2] = g; p[3] = 255;
            }
        }
        other => bail!("czi-rs: unsupported pixel type {other:?}"),
    }

    RgbaImage::from_raw(w, h, rgba).context("czi-rs: failed to build RgbaImage")
}

impl Slide for CziBackend {
    fn dimensions(&self) -> Result<(u64, u64)> { Ok((self.width, self.height)) }
    fn level_count(&self)  -> Result<u32>        { Ok(1) }
    fn level_dimensions(&self, _l: u32) -> Result<(u64, u64)> { Ok((self.width, self.height)) }
    fn level_downsample(&self, _l: u32) -> Result<f64>        { Ok(1.0) }
    fn best_level_for_downsample(&self, _ds: f64) -> Result<u32> { Ok(0) }

    fn read_region(&self, x: u64, y: u64, _level: u32, w: u32, h: u32) -> Result<RgbaImage> {
        let (x, y)   = (x as u32, y as u32);
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

// ─── 3. openslide-pure-rs backend ────────────────────────────────────────────
//
// The SlideBackend trait inside openslide-pure-rs is pub(crate) and lacks
// Send+Sync bounds, so Box<dyn SlideBackend> does not auto-implement them.
// The concrete MiraxSlide backend uses only Arc + internal Mutex, making it
// genuinely thread-safe; we assert that here.

struct OsBackend {
    inner: openslide_pure_rs::OpenSlide,
    channel_count: u32,
}

// SAFETY: openslide-pure-rs's only concrete backend (MiraxSlide) uses
// Arc + Mutex internally.  All methods take &self and serialise I/O behind
// that Mutex, so sending and sharing across threads is safe.
unsafe impl Send for OsBackend {}
unsafe impl Sync for OsBackend {}

impl OsBackend {
    fn open(path: &Path) -> Result<Self> {
        let inner = openslide_pure_rs::OpenSlide::open(path)?;
        let channel_count = inner.channel_count();
        Ok(Self { inner, channel_count })
    }
}

impl Slide for OsBackend {
    fn dimensions(&self) -> Result<(u64, u64)> {
        self.inner.level_dimensions(0)
            .ok_or_else(|| anyhow::anyhow!("openslide-pure-rs: no level 0"))
    }

    fn level_count(&self) -> Result<u32> {
        Ok(self.inner.level_count())
    }

    fn level_dimensions(&self, level: u32) -> Result<(u64, u64)> {
        self.inner.level_dimensions(level)
            .ok_or_else(|| anyhow::anyhow!("openslide-pure-rs: level {level} not found"))
    }

    fn level_downsample(&self, level: u32) -> Result<f64> {
        self.inner.level_downsample(level)
            .ok_or_else(|| anyhow::anyhow!("openslide-pure-rs: level {level} not found"))
    }

    fn best_level_for_downsample(&self, downsample: f64) -> Result<u32> {
        Ok(self.inner.best_level_for_downsample(downsample))
    }

    fn read_region(&self, x: u64, y: u64, level: u32, w: u32, h: u32) -> Result<RgbaImage> {
        // Map logical channels to R/G/B(/A) based on channel count.
        let channels: [Option<u32>; 4] = match self.channel_count {
            1 => [Some(0), Some(0), Some(0), None],
            2 => [Some(0), Some(1), None,    None],
            3 => [Some(0), Some(1), Some(2), None],
            _ => [Some(0), Some(1), Some(2), Some(3)],
        };

        let img = self.inner.read_region_rgba(channels, x as i64, y as i64, level, w, h)?;

        // openslide_pure_rs::RgbaImage → image::RgbaImage (same memory layout)
        RgbaImage::from_raw(img.width, img.height, img.data)
            .context("openslide-pure-rs: failed to build RgbaImage")
    }
}
