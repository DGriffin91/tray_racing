use std::time::Duration;

use wgpu::*;

pub struct Timestamp {
    buffer: Buffer,
    readback: Buffer,
    queries: QuerySet,
    timestamp_period: f32,
}

impl Timestamp {
    pub fn new(device: &Device, queue: &Queue) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Timestamps buffer"),
            size: 16,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback = device.create_buffer(&BufferDescriptor {
            label: None,
            size: 16,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        readback.unmap();

        let queries = device.create_query_set(&QuerySetDescriptor {
            label: None,
            count: 2,
            ty: QueryType::Timestamp,
        });

        Timestamp {
            buffer,
            readback,
            queries,
            timestamp_period: queue.get_timestamp_period(),
        }
    }

    pub fn start(&self, cpass: &mut ComputePass<'_>) {
        cpass.write_timestamp(&self.queries, 0);
    }

    pub fn end(&self, cpass: &mut ComputePass<'_>) {
        cpass.write_timestamp(&self.queries, 1);
    }

    pub fn resolve(&self, encoder: &mut CommandEncoder) {
        encoder.resolve_query_set(&self.queries, 0..2, &self.buffer, 0);
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &self.readback, 0, self.buffer.size());
    }

    pub fn map(&self) -> BufferSlice<'_> {
        let timestamp_slice = self.readback.slice(..);
        timestamp_slice.map_async(MapMode::Read, |r| r.unwrap());
        timestamp_slice
    }

    pub fn unmap(&self, slice: BufferSlice<'_>) -> Duration {
        let timing_data = slice.get_mapped_range();
        let timings = timing_data
            .chunks_exact(8)
            .map(|b| u64::from_ne_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        drop(timing_data);
        self.readback.unmap();
        Duration::from_nanos(
            ((timings[1] - timings[0]) as f64 * f64::from(self.timestamp_period)) as u64,
        )
    }

    pub fn get_ms(&self, device: &Device) -> f32 {
        let timestamp_slice = self.map();
        device.poll(Maintain::Wait);
        let time_ms = self.unmap(timestamp_slice).as_secs_f32() * 1000.0;
        time_ms
    }
}
