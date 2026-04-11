/// Metadata describing a Deep Zoom Image pyramid.
#[derive(Clone, Debug)]
pub struct DziInfo {
    pub width: u64,
    pub height: u64,
    pub tile_size: u32,
    pub overlap: u32,
    /// Number of DZI levels (level 0 = 1×1, level max_level = full resolution).
    pub max_level: u32,
}

/// Pixel bounds of one DZI tile inside a level's virtual canvas.
#[derive(Debug)]
pub struct TileBounds {
    /// Top-left x in the level canvas (pixels, including overlap).
    pub x0: u64,
    /// Top-left y in the level canvas (pixels, including overlap).
    pub y0: u64,
    /// Tile width  (may be smaller at right/bottom edge).
    pub w: u32,
    /// Tile height (may be smaller at right/bottom edge).
    pub h: u32,
}

impl DziInfo {
    /// Build a DziInfo for a slide of `width × height` at level 0.
    pub fn new(width: u64, height: u64, tile_size: u32, overlap: u32) -> Self {
        let max_dim = width.max(height);
        // Number of levels = ceil(log2(max_dim)) + 1, minimum 1.
        let max_level = if max_dim <= 1 {
            0
        } else {
            (max_dim as f64).log2().ceil() as u32
        };
        Self { width, height, tile_size, overlap, max_level }
    }

    /// Dimensions (w, h) of the virtual canvas at the given DZI level.
    pub fn level_canvas(&self, level: u32) -> (u64, u64) {
        let shift = self.max_level.saturating_sub(level);
        let scale: u64 = 1 << shift;
        let w = ((self.width + scale - 1) / scale).max(1);
        let h = ((self.height + scale - 1) / scale).max(1);
        (w, h)
    }

    /// Downscale factor of a DZI level relative to level 0 (full resolution).
    /// level == max_level ⟹ 1.0 (no downscale).
    pub fn level_downsample(&self, level: u32) -> f64 {
        let shift = self.max_level.saturating_sub(level);
        (1u64 << shift) as f64
    }

    /// Pixel bounds (with overlap) of tile (col, row) at the given DZI level.
    pub fn tile_bounds(&self, level: u32, col: u32, row: u32) -> TileBounds {
        let (canvas_w, canvas_h) = self.level_canvas(level);

        // Left edge of the inner (non-overlapping) tile region.
        let inner_x = col * self.tile_size;
        let inner_y = row * self.tile_size;

        // Extend by overlap on all sides, clamped to canvas.
        let x0 = inner_x.saturating_sub(self.overlap) as u64;
        let y0 = inner_y.saturating_sub(self.overlap) as u64;
        let x1 = ((inner_x + self.tile_size + self.overlap) as u64).min(canvas_w);
        let y1 = ((inner_y + self.tile_size + self.overlap) as u64).min(canvas_h);

        TileBounds {
            x0,
            y0,
            w: (x1 - x0) as u32,
            h: (y1 - y0) as u32,
        }
    }

    /// Render the DZI XML descriptor (as expected by OpenSeadragon).
    pub fn to_xml(&self, format: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
       Format="{format}"
       Overlap="{overlap}"
       TileSize="{tile_size}">
  <Size Width="{width}" Height="{height}"/>
</Image>"#,
            format = format,
            overlap = self.overlap,
            tile_size = self.tile_size,
            width = self.width,
            height = self.height,
        )
    }
}
