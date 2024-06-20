pub mod bvh_embree;
pub mod bvh_embree_to_cwbvh;
pub mod embree_managed;
pub mod gpu_bvh_builder_embree;
pub mod gpu_bvh_builder_embree_bvh2;

pub fn new_embree_device(threads: usize, verbose: bool, limit_to_sse2: bool) -> embree4_rs::Device {
    embree4_rs::Device::try_new(Some(&format!(
        "threads={}{}{}\0",
        threads,
        if limit_to_sse2 { ",isa=sse2" } else { "" },
        if verbose { ",verbose=4" } else { "" }
    )))
    .unwrap()
}

pub unsafe extern "C" fn embree_error_fn(
    _user_ptr: *mut ::std::os::raw::c_void,
    code: embree4_sys::RTCError,
    str_: *const ::std::os::raw::c_char,
) {
    panic!(
        "(GPU) Embree error: {code:?} {:?}",
        std::ffi::CStr::from_ptr(str_)
    );
}
