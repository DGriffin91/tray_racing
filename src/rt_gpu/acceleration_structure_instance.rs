use bytemuck::{Pod, Zeroable};
use glam::Affine3A;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AccelerationStructureInstance {
    pub transform: [f32; 12],
    pub custom_index_and_mask: u32,
    pub shader_binding_table_record_offset_and_flags: u32,
    pub acceleration_structure_reference: u64,
}

unsafe impl Pod for AccelerationStructureInstance {}
unsafe impl Zeroable for AccelerationStructureInstance {}

impl std::fmt::Debug for AccelerationStructureInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instance")
            .field("transform", &self.transform)
            .field("custom_index()", &self.custom_index())
            .field("mask()", &self.mask())
            .field(
                "shader_binding_table_record_offset()",
                &self.shader_binding_table_record_offset(),
            )
            .field("flags()", &self.flags())
            .field(
                "acceleration_structure_reference",
                &self.acceleration_structure_reference,
            )
            .finish()
    }
}

#[allow(dead_code)]
impl AccelerationStructureInstance {
    const LOW_24_MASK: u32 = 0x00ff_ffff;
    const MAX_U24: u32 = (1u32 << 24u32) - 1u32;

    #[inline]
    pub fn affine_to_rows(mat: &Affine3A) -> [f32; 12] {
        let row_0 = mat.matrix3.row(0);
        let row_1 = mat.matrix3.row(1);
        let row_2 = mat.matrix3.row(2);
        let translation = mat.translation;
        [
            row_0.x,
            row_0.y,
            row_0.z,
            translation.x,
            row_1.x,
            row_1.y,
            row_1.z,
            translation.y,
            row_2.x,
            row_2.y,
            row_2.z,
            translation.z,
        ]
    }

    #[inline]
    pub fn rows_to_affine(rows: &[f32; 12]) -> Affine3A {
        Affine3A::from_cols_array(&[
            rows[0], rows[3], rows[6], rows[9], rows[1], rows[4], rows[7], rows[10], rows[2],
            rows[5], rows[8], rows[11],
        ])
    }

    pub fn transform_as_affine(&self) -> Affine3A {
        Self::rows_to_affine(&self.transform)
    }
    pub fn set_transform(&mut self, transform: &Affine3A) {
        self.transform = Self::affine_to_rows(transform);
    }

    pub fn custom_index(&self) -> u32 {
        self.custom_index_and_mask & Self::LOW_24_MASK
    }

    pub fn mask(&self) -> u8 {
        (self.custom_index_and_mask >> 24) as u8
    }

    pub fn shader_binding_table_record_offset(&self) -> u32 {
        self.shader_binding_table_record_offset_and_flags & Self::LOW_24_MASK
    }

    pub fn flags(&self) -> u8 {
        (self.shader_binding_table_record_offset_and_flags >> 24) as u8
    }

    pub fn set_custom_index(&mut self, custom_index: u32) {
        debug_assert!(
            custom_index <= Self::MAX_U24,
            "custom_index uses more than 24 bits! {custom_index} > {}",
            Self::MAX_U24
        );
        self.custom_index_and_mask =
            (custom_index & Self::LOW_24_MASK) | (self.custom_index_and_mask & !Self::LOW_24_MASK)
    }

    pub fn set_mask(&mut self, mask: u8) {
        self.custom_index_and_mask =
            (self.custom_index_and_mask & Self::LOW_24_MASK) | (u32::from(mask) << 24)
    }

    pub fn set_shader_binding_table_record_offset(
        &mut self,
        shader_binding_table_record_offset: u32,
    ) {
        debug_assert!(shader_binding_table_record_offset <= Self::MAX_U24, "shader_binding_table_record_offset uses more than 24 bits! {shader_binding_table_record_offset} > {}", Self::MAX_U24);
        self.shader_binding_table_record_offset_and_flags = (shader_binding_table_record_offset
            & Self::LOW_24_MASK)
            | (self.shader_binding_table_record_offset_and_flags & !Self::LOW_24_MASK)
    }

    pub fn set_flags(&mut self, flags: u8) {
        self.shader_binding_table_record_offset_and_flags =
            (self.shader_binding_table_record_offset_and_flags & Self::LOW_24_MASK)
                | (u32::from(flags) << 24)
    }

    pub fn new(
        transform: &Affine3A,
        custom_index: u32,
        mask: u8,
        shader_binding_table_record_offset: u32,
        flags: u8,
        acceleration_structure_reference: u64,
    ) -> Self {
        debug_assert!(
            custom_index <= Self::MAX_U24,
            "custom_index uses more than 24 bits! {custom_index} > {}",
            Self::MAX_U24
        );
        debug_assert!(
            shader_binding_table_record_offset <= Self::MAX_U24,
            "shader_binding_table_record_offset uses more than 24 bits! {shader_binding_table_record_offset} > {}", Self::MAX_U24
        );
        AccelerationStructureInstance {
            transform: Self::affine_to_rows(transform),
            custom_index_and_mask: (custom_index & Self::MAX_U24) | (u32::from(mask) << 24),
            shader_binding_table_record_offset_and_flags: (shader_binding_table_record_offset
                & Self::MAX_U24)
                | (u32::from(flags) << 24),
            acceleration_structure_reference,
        }
    }
}
