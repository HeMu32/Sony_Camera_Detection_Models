"""Integer reference for the V8 tensor processor's uint8 requantization."""


def nearest_even(numerator, denominator):
    low, remainder = divmod(numerator, denominator)
    return low + (2 * remainder > denominator or
                  (2 * remainder == denominator and low % 2 != 0))


def round_significand(numerator):
    step = 1 << max(0, abs(numerator).bit_length() - 16)
    return nearest_even(numerator, step) * step


def integer_formula(centered, output_zp, multiplier, shift):
    denominator = 1 << shift
    product = round_significand(centered * multiplier)
    biased = round_significand(product + output_zp * denominator)
    return max(0, min(255, nearest_even(biased, denominator)))
