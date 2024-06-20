use tracing::field::{Field, Visit};
use tracing::Event;
use tracing::Subscriber;
use tracing_subscriber::fmt::{self, format::Writer};
use tracing_subscriber::{
    fmt::format::{FormatEvent, FormatFields},
    util::SubscriberInitExt,
};
use tracing_subscriber::{layer::SubscriberExt, registry::LookupSpan};

struct CustomFormatter;

impl<S, N> FormatEvent<S, N> for CustomFormatter
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &fmt::FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> std::fmt::Result {
        let mut visitor = TimingVisitor::default();
        event.record(&mut visitor);

        // Mostly to filter out wgpu
        let Some(module_path) = event.metadata().module_path() else {
            return Ok(());
        };
        if !module_path.contains("obvhs::") {
            // TODO see if there's a better way to filter
            return Ok(());
        }

        let mut span_ids = vec![];
        if let Some(span) = ctx.lookup_current() {
            span_ids.push(span.id());
            let mut parent_span = span.parent();
            while let Some(parent) = parent_span {
                span_ids.push(parent.id());
                parent_span = parent.parent();
            }
        }

        write!(writer, "{:8}", visitor.time)?;

        for _ in 0..span_ids.len() {
            write!(writer, " ")?;
        }
        if span_ids.len() > 0 {
            write!(writer, " / ")?;
        } else {
            write!(writer, " | ")?;
        }
        write!(writer, "{}", event.metadata().name())?;
        //for span_id in span_ids.iter().rev() {
        //    if let Some(span) = ctx.span(span_id) {
        //        write!(writer, " < {}", span.name())?;
        //        if let Some(fields) = span.extensions().get::<fmt::FormattedFields<()>>() {
        //            write!(writer, "{{{}}}", fields)?;
        //        }
        //    }
        //}
        writeln!(writer)?;

        Ok(())
    }
}
#[derive(Default)]
struct TimingVisitor {
    time: String,
}

impl Visit for TimingVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "time.busy" {
            self.time = format!("{:?}", value);
        }
    }
}

pub fn setup_subscriber() {
    //tracing_subscriber::fmt::fmt()
    //    .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
    //    .with_target(false)
    //    .with_level(false)
    //    .init();

    //let my_filter = filter::filter_fn(|metadata| metadata.target().contains("obvhs_verbose"));
    let layer = tracing_subscriber::fmt::layer()
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        .event_format(CustomFormatter);
    tracing_subscriber::registry()
        .with(layer) //.with_filter(my_filter)
        .init();
}
