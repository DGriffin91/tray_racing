use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    *,
};

pub fn fsampler_layout(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Sampler(SamplerBindingType::Filtering),
        count: None,
    }
}

pub fn csampler_layout(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Sampler(SamplerBindingType::Comparison),
        count: None,
    }
}

pub fn nsampler_layout(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
        count: None,
    }
}

pub fn texture_layout(
    binding: u32,
    dim: TextureViewDimension,
    sample_type: TextureSampleType,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Texture {
            sample_type,
            view_dimension: dim,
            multisampled: false,
        },
        count: None,
    }
}

pub fn rstorage_texture_layout(
    binding: u32,
    dim: TextureViewDimension,
    format: TextureFormat,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
            access: StorageTextureAccess::ReadOnly,
            format,
            view_dimension: dim,
        },
        count: None,
    }
}

pub fn wstorage_texture_layout(
    binding: u32,
    dim: TextureViewDimension,
    format: TextureFormat,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format,
            view_dimension: dim,
        },
        count: None,
    }
}

pub fn rwstorage_texture_layout(
    binding: u32,
    dim: TextureViewDimension,
    format: TextureFormat,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
            access: StorageTextureAccess::ReadWrite,
            format,
            view_dimension: dim,
        },
        count: None,
    }
}

pub fn ftexture_layout(binding: u32, dim: TextureViewDimension) -> BindGroupLayoutEntry {
    texture_layout(binding, dim, TextureSampleType::Float { filterable: true })
}

pub fn dtexture_layout(binding: u32, dim: TextureViewDimension) -> BindGroupLayoutEntry {
    texture_layout(binding, dim, TextureSampleType::Depth)
}

pub fn utexture_layout(binding: u32, dim: TextureViewDimension) -> BindGroupLayoutEntry {
    texture_layout(binding, dim, TextureSampleType::Uint)
}

pub fn stexture_layout(binding: u32, dim: TextureViewDimension) -> BindGroupLayoutEntry {
    texture_layout(binding, dim, TextureSampleType::Sint)
}

pub fn uniform_layout(binding: u32, min_size: std::num::NonZeroU64) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(min_size),
        },
        count: None,
    }
}

pub fn rw_storage_buffer_layout(
    binding: u32,
    min_size: std::num::NonZeroU64,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            has_dynamic_offset: false,
            min_binding_size: Some(min_size),
            ty: BufferBindingType::Storage { read_only: false },
        },
        count: None,
    }
}

pub fn storage_buffer_layout(binding: u32, min_size: std::num::NonZeroU64) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            has_dynamic_offset: false,
            min_binding_size: Some(min_size),
            ty: BufferBindingType::Storage { read_only: true },
        },
        count: None,
    }
}

pub fn uniform_buffer(data: &[u8], device: &Device, label: &str) -> Buffer {
    let config_uniform = device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: data.as_ref(),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });
    config_uniform
}

pub fn opaque_target(format: TextureFormat) -> Option<ColorTargetState> {
    Some(ColorTargetState {
        format,
        blend: None,
        write_mask: ColorWrites::ALL,
    })
}

pub fn load_color_attachment(view: &TextureView) -> Option<RenderPassColorAttachment<'_>> {
    Some(RenderPassColorAttachment {
        view,
        resolve_target: None,
        ops: Operations {
            load: LoadOp::Load,
            store: StoreOp::Store,
        },
    })
}

pub fn clear_color_attachment(view: &TextureView) -> Option<RenderPassColorAttachment<'_>> {
    Some(RenderPassColorAttachment {
        view,
        resolve_target: None,
        ops: Operations {
            load: LoadOp::Clear(Default::default()),
            store: StoreOp::Store,
        },
    })
}

pub fn load_depth_attachment(view: &TextureView) -> Option<RenderPassDepthStencilAttachment<'_>> {
    Some(RenderPassDepthStencilAttachment {
        view,
        depth_ops: Some(Operations {
            load: LoadOp::Load,
            store: StoreOp::Store,
        }),
        stencil_ops: None,
    })
}

pub fn init_storage(label: &str, device: &Device, bytes: &[u8]) -> Buffer {
    let bvh_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    });
    bvh_buffer
}
