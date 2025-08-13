use crate::error::FixedPointError;

#[inline]
pub(crate) fn extract_from_raw_bytes<T>(bytes: &[u8], range: std::ops::Range<usize>) -> Result<T, FixedPointError>
where 
    T: Sized + for<'a> TryFrom<&'a [u8]>,
{
    T::try_from(&bytes[range]).map_err(|_| FixedPointError::RangeError)
}