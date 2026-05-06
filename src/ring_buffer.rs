/// Fixed-capacity ring buffer for out-of-order inserts and in-order draining.
///
/// Items are placed by their monotonic `id` into slot `id % capacity`. `drain` yields
/// consecutive items starting at the lowest unseen id, stopping as soon as a gap is found.
///
/// The caller must guarantee that no two live ids share the same slot, i.e. ids `X` and
/// `X + capacity` are never both in the buffer simultaneously.
pub struct RingBuffer<T> {
    capacity: usize,
    slots: Vec<Option<T>>,
    next: usize,
}

impl<T> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        let slots = (0..capacity).map(|_| None).collect();
        Self {
            capacity,
            slots,
            next: 0,
        }
    }

    /// Store `item` at the slot for `id`. Silently overwrites if slot `id % capacity` is already occupied
    pub fn insert(&mut self, id: usize, item: T) {
        let idx = id % self.capacity;
        self.slots[idx] = Some(item);
    }

    /// Yield consecutive items starting from the next expected id, stopping at the first gap.
    pub fn drain(&mut self) -> impl Iterator<Item = T> + '_ {
        std::iter::from_fn(move || {
            let slot = &mut self.slots[self.next % self.capacity];
            slot.take().inspect(|_| {
                self.next += 1;
            })
        })
    }
}
