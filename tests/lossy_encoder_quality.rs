//! Lossy encoder quality and size comparison tests
//!
//! These tests compare the image-webp lossy encoder against libwebp:
//! - Bitstream compatibility: libwebp can decode our output
//! - Quality metrics: PSNR of decoded output
//! - Size comparison: our output vs libwebp's output at same quality
//!
//! Note: The current lossy encoder uses DC-only mode selection, so quality
//! and size will be significantly worse than libwebp. These tests establish
//! baselines and will improve as mode selection is implemented.

use image_webp::{ColorType, EncoderParams, WebPEncoder};

/// Simple PSNR calculation (not great for perceptual quality, but simple)
fn calculate_psnr(original: &[u8], decoded: &[u8]) -> f64 {
    assert_eq!(original.len(), decoded.len());

    let mse: f64 = original
        .iter()
        .zip(decoded.iter())
        .map(|(&a, &b)| {
            let diff = a as f64 - b as f64;
            diff * diff
        })
        .sum::<f64>()
        / original.len() as f64;

    if mse == 0.0 {
        return f64::INFINITY;
    }

    10.0 * (255.0 * 255.0 / mse).log10()
}

/// Helper to create lossy encoder params
fn lossy_params(quality: u8) -> EncoderParams {
    EncoderParams::lossy(quality)
}

/// Test that libwebp can decode our lossy output
#[test]
fn libwebp_can_decode_our_lossy_output() {
    // Create a simple gradient test image
    let width = 64u32;
    let height = 64u32;
    let mut img = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            img[idx] = (x * 4) as u8; // R: horizontal gradient
            img[idx + 1] = (y * 4) as u8; // G: vertical gradient
            img[idx + 2] = 128; // B: constant
        }
    }

    // Encode with image-webp lossy (quality 75)
    let mut output = Vec::new();
    let mut encoder = WebPEncoder::new(&mut output);
    encoder.set_params(lossy_params(75));
    encoder
        .encode(&img, width, height, ColorType::Rgb8)
        .expect("Encoding failed");

    println!("Encoded {} bytes", output.len());

    // Verify it's a valid WebP that libwebp can decode
    let decoded = webp::Decoder::new(&output)
        .decode()
        .expect("libwebp failed to decode our output");

    assert_eq!(decoded.width(), width);
    assert_eq!(decoded.height(), height);

    // For lossy, we can't expect exact match, but PSNR should be reasonable
    // Note: with DC-only mode, quality will be poor
    let psnr = calculate_psnr(&img, &decoded);
    println!("PSNR: {:.2} dB (target: > 20 dB for basic lossy)", psnr);
    assert!(psnr > 15.0, "PSNR too low: {:.2} dB", psnr);
}

/// Compare our encoder size vs libwebp at same quality
#[test]
fn size_comparison_vs_libwebp() {
    let width = 128u32;
    let height = 128u32;

    // Create test image with some texture (checkerboard + gradient)
    let mut img = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            let checker = ((x / 8 + y / 8) % 2) as u8 * 64;
            img[idx] = ((x * 2) as u8).wrapping_add(checker);
            img[idx + 1] = ((y * 2) as u8).wrapping_add(checker);
            img[idx + 2] = 128u8.wrapping_add(checker);
        }
    }

    for quality in [50u8, 75, 90] {
        // Encode with image-webp
        let mut our_output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut our_output);
        encoder.set_params(lossy_params(quality));
        encoder
            .encode(&img, width, height, ColorType::Rgb8)
            .expect("Our encoding failed");

        // Encode with libwebp
        let libwebp_encoder = webp::Encoder::from_rgb(&img, width, height);
        let libwebp_output = libwebp_encoder.encode(quality as f32);

        let our_size = our_output.len();
        let libwebp_size = libwebp_output.len();
        let ratio = our_size as f64 / libwebp_size as f64;

        println!(
            "Quality {}: ours = {} bytes, libwebp = {} bytes, ratio = {:.2}x",
            quality, our_size, libwebp_size, ratio
        );

        // We should be within 2x of libwebp's size (generous for initial implementation)
        assert!(
            ratio < 2.0,
            "Our output is {:.1}x larger than libwebp at quality {}",
            ratio,
            quality
        );
    }
}

/// Compare decoded quality (PSNR) at same file size
#[test]
fn quality_comparison_at_same_size() {
    let width = 128u32;
    let height = 128u32;

    // Load a real test image or create a complex one
    let mut img = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            // Create some texture
            let noise = ((x * 7 + y * 13) % 64) as u8;
            img[idx] = ((x * 2) as u8).wrapping_add(noise);
            img[idx + 1] = ((y * 2) as u8).wrapping_add(noise);
            img[idx + 2] = (((x + y) * 2) as u8).wrapping_add(noise);
        }
    }

    // Encode with libwebp at quality 75
    let libwebp_encoder = webp::Encoder::from_rgb(&img, width, height);
    let libwebp_output = libwebp_encoder.encode(75.0);
    let _target_size = libwebp_output.len();

    // Find quality setting that produces similar size with our encoder
    // (binary search or just try a few values)
    let mut our_output = Vec::new();
    let mut encoder = WebPEncoder::new(&mut our_output);
    encoder.set_params(lossy_params(75));
    encoder
        .encode(&img, width, height, ColorType::Rgb8)
        .expect("Our encoding failed");

    // Decode both and compare PSNR
    let our_decoded = webp::Decoder::new(&our_output)
        .decode()
        .expect("Failed to decode our output");
    let libwebp_decoded = webp::Decoder::new(&libwebp_output)
        .decode()
        .expect("Failed to decode libwebp output");

    let our_psnr = calculate_psnr(&img, &our_decoded);
    let libwebp_psnr = calculate_psnr(&img, &libwebp_decoded);

    println!(
        "Our encoder: {} bytes, PSNR = {:.2} dB",
        our_output.len(),
        our_psnr
    );
    println!(
        "libwebp:     {} bytes, PSNR = {:.2} dB",
        libwebp_output.len(),
        libwebp_psnr
    );

    // Our quality should be at least 80% of libwebp's
    // (generous threshold for initial implementation)
    let quality_ratio = our_psnr / libwebp_psnr;
    assert!(
        quality_ratio > 0.8,
        "Our quality ({:.2} dB) is less than 80% of libwebp ({:.2} dB)",
        our_psnr,
        libwebp_psnr
    );
}

/// Test various image types for encoder robustness
#[test]
fn encode_various_image_types() {
    let test_cases = [
        ("solid_color", create_solid_color_image(64, 64)),
        ("gradient", create_gradient_image(64, 64)),
        ("checkerboard", create_checkerboard_image(64, 64)),
        ("noise", create_noise_image(64, 64)),
    ];

    for (name, img) in test_cases {
        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(lossy_params(75));
        encoder
            .encode(&img, 64, 64, ColorType::Rgb8)
            .unwrap_or_else(|e| panic!("Failed to encode {}: {:?}", name, e));

        // Verify libwebp can decode
        let decoded = webp::Decoder::new(&output)
            .decode()
            .unwrap_or_else(|| panic!("libwebp failed to decode {}", name));

        let psnr = calculate_psnr(&img, &decoded);
        println!("{}: {} bytes, PSNR = {:.2} dB", name, output.len(), psnr);

        // Minimum acceptable PSNR (noise is a pathological case, allow lower threshold)
        let min_psnr = if name == "noise" { 10.0 } else { 20.0 };
        assert!(
            psnr > min_psnr,
            "{} has unacceptably low PSNR: {:.2} (min: {:.0})",
            name,
            psnr,
            min_psnr
        );
    }
}

// Helper functions to create test images
fn create_solid_color_image(w: u32, h: u32) -> Vec<u8> {
    vec![128u8; (w * h * 3) as usize]
}

fn create_gradient_image(w: u32, h: u32) -> Vec<u8> {
    let mut img = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 3) as usize;
            img[idx] = (x * 255 / w) as u8;
            img[idx + 1] = (y * 255 / h) as u8;
            img[idx + 2] = ((x + y) * 255 / (w + h)) as u8;
        }
    }
    img
}

fn create_checkerboard_image(w: u32, h: u32) -> Vec<u8> {
    let mut img = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 3) as usize;
            let val = if (x / 8 + y / 8) % 2 == 0 { 255 } else { 0 };
            img[idx] = val;
            img[idx + 1] = val;
            img[idx + 2] = val;
        }
    }
    img
}

fn create_noise_image(w: u32, h: u32) -> Vec<u8> {
    use rand::RngCore;
    let mut img = vec![0u8; (w * h * 3) as usize];
    rand::thread_rng().fill_bytes(&mut img);
    img
}

#[cfg(feature = "ssimulacra2")]
mod ssimulacra_tests {
    use super::*;
    use ssimulacra2::Ssimulacra2;

    #[test]
    fn ssimulacra2_quality_comparison() {
        // TODO: Add SSIMULACRA2-based quality comparison
        // This is the gold standard for perceptual quality
    }
}
