use std::{io::Read, path::Path, process::Command};

pub fn load_shader_module(path: &Path) -> Vec<u8> {
    let mut f = std::fs::File::open(&path).expect("no file found");
    let metadata = std::fs::metadata(&path).expect("unable to read metadata");
    let mut buffer = vec![0; metadata.len() as usize];
    f.read(&mut buffer).expect("buffer overflow");
    buffer
}

pub fn compile_to_spirv(src_path: &str, dst_string: &str, profile: &str) {
    let out = Command::new("dxc")
        .arg(src_path)
        .args(["-O3", "-T", profile, "-spirv", "-Fo", dst_string])
        .output()
        .expect("failed to execute process");

    if out.stderr.len() > 1 {
        println!("dxc stderr: {}", String::from_utf8_lossy(&out.stderr));
    }
}
