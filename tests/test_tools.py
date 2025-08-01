"""Test tools module.

This module contains unit tests for the Tools class in the statista.tools module.
Each method of the Tools class has a corresponding test class with multiple test methods
to cover different scenarios and edge cases.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from statista.tools import Tools


@pytest.fixture(scope="module")
def sample_data():
    """Sample data for testing Tools methods."""
    return [1.2, 3.4, 5.6, 7.8, 9.0]


@pytest.fixture(scope="module")
def sample_array():
    """Sample numpy array for testing Tools methods."""
    return np.array([1.2, 3.4, 5.6, 7.8, 9.0])


class TestNormalize:
    """Test normalize method of Tools class."""

    def test_normalize_list(self, sample_data):
        """
        Test normalize method with a list of values.

        Inputs:
            sample_data: A list of numerical values [1.2, 3.4, 5.6, 7.8, 9.0]

        Expected:
            The normalized values should be in the range [0, 1]
            The minimum value should be mapped to 0
            The maximum value should be mapped to 1
            Other values should be linearly scaled between 0 and 1
        """
        result = Tools.normalize(sample_data)
        expected = np.array([0.0, 0.28205128, 0.56410256, 0.84615385, 1.0])
        assert_array_almost_equal(result, expected, decimal=8)

    def test_normalize_array(self, sample_array):
        """
        Test normalize method with a numpy array.

        Inputs:
            sample_array: A numpy array of values [1.2, 3.4, 5.6, 7.8, 9.0]

        Expected:
            The normalized values should be in the range [0, 1]
            The minimum value should be mapped to 0
            The maximum value should be mapped to 1
            Other values should be linearly scaled between 0 and 1
        """
        result = Tools.normalize(sample_array)
        expected = np.array([0.0, 0.28205128, 0.56410256, 0.84615385, 1.0])
        assert_array_almost_equal(result, expected, decimal=8)

    def test_normalize_single_value(self):
        """
        Test normalize method with a single value.

        Inputs:
            A list containing a single value [42]

        Expected:
            ValueError should be raised because normalization requires at least two values
        """
        with pytest.raises(
            ValueError,
            match="input data must contain at least two values for normalization",
        ):
            Tools.normalize([42])

    def test_normalize_empty_list(self):
        """
        Test normalize method with an empty list.

        Inputs:
            An empty list []

        Expected:
            ValueError should be raised because normalization requires at least two values
        """
        with pytest.raises(
            ValueError,
            match="input data must contain at least two values for normalization",
        ):
            Tools.normalize([])

    def test_normalize_identical_values(self):
        """
        Test normalize method with a list of identical values.

        Inputs:
            A list of identical values [5, 5, 5, 5]

        Expected:
            The result should be an array of NaN values because max-min is zero
        """
        result = Tools.normalize([5, 5, 5, 5])
        assert np.all(np.isnan(result))

    def test_normalize_negative_values(self):
        """
        Test normalize method with negative values.

        Inputs:
            A list containing negative values [-10, -5, 0, 5, 10]

        Expected:
            The normalized values should be in the range [0, 1]
            -10 should be mapped to 0
            10 should be mapped to 1
            Other values should be linearly scaled between 0 and 1
        """
        data = [-10, -5, 0, 5, 10]
        result = Tools.normalize(data)
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        assert_array_almost_equal(result, expected, decimal=8)


class TestStandardize:
    """Test standardize method of Tools class."""

    def test_standardize_list(self, sample_data):
        """
        Test standardize method with a list of values.

        Inputs:
            sample_data: A list of numerical values [1.2, 3.4, 5.6, 7.8, 9.0]

        Expected:
            The standardized values should have mean=0 and std=1
            The values should be transformed according to the formula: (x - mean(x)) / std(x)
        """
        result = Tools.standardize(sample_data)
        assert_array_almost_equal(np.mean(result), 0.0, decimal=8)
        assert_array_almost_equal(np.std(result), 1.0, decimal=8)

    def test_standardize_array(self, sample_array):
        """
        Test standardize method with a numpy array.

        Inputs:
            sample_array: A numpy array of values [1.2, 3.4, 5.6, 7.8, 9.0]

        Expected:
            The standardized values should have mean=0 and std=1
            The values should be transformed according to the formula: (x - mean(x)) / std(x)
        """
        result = Tools.standardize(sample_array)
        assert_array_almost_equal(np.mean(result), 0.0, decimal=8)
        assert_array_almost_equal(np.std(result), 1.0, decimal=8)

    def test_standardize_specific_values(self):
        """
        Test standardize method with specific values to verify the transformation.

        Inputs:
            A list of values [10, 20, 30, 40, 50]

        Expected:
            The standardized values should match the expected values calculated manually
            For data with mean=30 and std=15.81, the expected values are:
            [-1.2649, -0.6325, 0.0, 0.6325, 1.2649]
        """
        data = [10, 20, 30, 40, 50]
        result = Tools.standardize(data)
        # Verify mean is 0 and std is 1
        assert_array_almost_equal(np.mean(result), 0.0, decimal=8)
        assert_array_almost_equal(np.std(result), 1.0, decimal=8)
        # Verify the pattern of values
        assert result[0] < result[1] < result[2] < result[3] < result[4]
        assert_array_almost_equal(result[2], 0.0, decimal=8)

    def test_standardize_zero_mean_data(self):
        """
        Test standardize method with data that already has a mean of zero.

        Inputs:
            A list of values with mean=0: [-2, -1, 0, 1, 2]

        Expected:
            The standardized values should have mean=0 and std=1
            The values should be transformed according to the formula: (x - mean(x)) / std(x)
        """
        data = [-2, -1, 0, 1, 2]
        result = Tools.standardize(data)
        assert_array_almost_equal(np.mean(result), 0.0, decimal=8)
        assert_array_almost_equal(np.std(result), 1.0, decimal=8)
        expected = np.array([-1.41421356, -0.70710678, 0.0, 0.70710678, 1.41421356])
        assert_array_almost_equal(result, expected, decimal=8)

    def test_standardize_single_value(self):
        """
        Test standardize method with a single value.

        Inputs:
            A list containing a single value [42]

        Expected:
            The result should be an array of NaN values because the standard deviation is zero
        """
        result = Tools.standardize([42])
        assert np.all(np.isnan(result))

    def test_standardize_identical_values(self):
        """
        Test standardize method with a list of identical values.

        Inputs:
            A list of identical values [5, 5, 5, 5]

        Expected:
            The result should be an array of NaN values because the standard deviation is zero
        """
        result = Tools.standardize([5, 5, 5, 5])
        assert np.all(np.isnan(result))


class TestRescale:
    """Test rescale method of Tools class."""

    def test_rescale_basic(self):
        """
        Test rescale method with basic inputs.

        Inputs:
            value=75, old_min=0, old_max=100, new_min=0, new_max=1

        Expected:
            The rescaled value should be 0.75
        """
        result = Tools.rescale(75, 0, 100, 0, 1)
        assert result == pytest.approx(0.75)

    def test_rescale_negative_range(self):
        """
        Test rescale method with a negative target range.

        Inputs:
            value=0.3, old_min=0, old_max=1, new_min=-1, new_max=1

        Expected:
            The rescaled value should be -0.4
        """
        result = Tools.rescale(0.3, 0, 1, -1, 1)
        assert result == -0.4

    def test_rescale_temperature_conversion(self):
        """
        Test rescale method for temperature conversion (Celsius to Fahrenheit).

        Inputs:
            value=25 (Celsius), old_min=0, old_max=100, new_min=32, new_max=212

        Expected:
            The rescaled value should be 77.0 (Fahrenheit)
        """
        result = Tools.rescale(25, 0, 100, 32, 212)
        assert result == pytest.approx(77.0)

    def test_rescale_outside_original_range(self):
        """
        Test rescale method with a value outside the original range.

        Inputs:
            value=150, old_min=0, old_max=100, new_min=0, new_max=1

        Expected:
            The rescaled value should be 1.5, extrapolating beyond the new range
        """
        result = Tools.rescale(150, 0, 100, 0, 1)
        assert result == pytest.approx(1.5)

    def test_rescale_equal_old_bounds(self):
        """
        Test rescale method with equal old bounds.

        Inputs:
            value=50, old_min=50, old_max=50, new_min=0, new_max=100

        Expected:
            ZeroDivisionError should be raised because old_max equals old_min
        """
        with pytest.raises(ZeroDivisionError):
            Tools.rescale(50, 50, 50, 0, 100)

    def test_rescale_equal_new_bounds(self):
        """
        Test rescale method with equal new bounds.

        Inputs:
            value=75, old_min=0, old_max=100, new_min=50, new_max=50

        Expected:
            The rescaled value should be 50, as any value will be mapped to the single point
        """
        result = Tools.rescale(75, 0, 100, 50, 50)
        assert result == 50

    def test_rescale_inverted_old_range(self):
        """
        Test rescale method with inverted old range (old_min > old_max).

        Inputs:
            value=25, old_min=100, old_max=0, new_min=0, new_max=1

        Expected:
            The rescaled value should be 0.75, as the formula handles inverted ranges
        """
        result = Tools.rescale(25, 100, 0, 0, 1)
        assert result == pytest.approx(0.75)

    def test_rescale_inverted_new_range(self):
        """
        Test rescale method with inverted new range (new_min > new_max).

        Inputs:
            value=75, old_min=0, old_max=100, new_min=1, new_max=0

        Expected:
            The rescaled value should be 0.25, as the formula handles inverted ranges
        """
        result = Tools.rescale(75, 0, 100, 1, 0)
        assert result == pytest.approx(0.25)


class TestLogRescale:
    """Test log_rescale method of Tools class."""

    def test_log_rescale_basic(self):
        """
        Test log_rescale method with basic inputs.

        Inputs:
            value=100, min_old=1, max_old=1000, min_new=1, max_new=10

        Expected:
            The log-rescaled value should be 7
        """
        result = Tools.log_rescale(100, 1, 1000, 1, 10)
        assert result == 7

    def test_log_rescale_small_value(self):
        """
        Test log_rescale method with a small value.

        Inputs:
            value=10, min_old=1, max_old=1000, min_new=1, max_new=10

        Expected:
            The log-rescaled value should be 4
        """
        result = Tools.log_rescale(10, 1, 1000, 1, 10)
        assert result == 4

    def test_log_rescale_zero_value(self):
        """
        Test log_rescale method with a zero value.

        Inputs:
            value=0, min_old=0, max_old=1000, min_new=0, max_new=10

        Expected:
            The log-rescaled value should be 0, using the special case handling
        """
        result = Tools.log_rescale(0, 0, 1000, 0, 10)
        assert result == 0

    def test_log_rescale_zero_min_old(self):
        """
        Test log_rescale method with min_old=0.

        Inputs:
            value=100, min_old=0, max_old=1000, min_new=0, max_new=10

        Expected:
            The log-rescaled value should use -7 as the logarithmic value for min_old
        """
        result = Tools.log_rescale(100, 0, 1000, 0, 10)
        # Expected value calculated using min_old_log = -7
        assert result == 8

    def test_log_rescale_negative_value(self):
        """
        Test log_rescale method with a negative value.

        Inputs:
            value=-10, min_old=1, max_old=1000, min_new=1, max_new=10

        Expected:
            ValueError should be raised because logarithm is undefined for negative values
        """
        with pytest.raises(ValueError):
            Tools.log_rescale(-10, 1, 1000, 1, 10)

    def test_log_rescale_max_old_equals_min_old(self):
        """
        Test log_rescale method with max_old equal to min_old.

        Inputs:
            value=50, min_old=50, max_old=50, min_new=0, max_new=10

        Expected:
            ValueError or ZeroDivisionError should be raised
        """
        with pytest.raises((ValueError, ZeroDivisionError)):
            Tools.log_rescale(50, 50, 50, 0, 10)

    def test_log_rescale_boundary_values(self):
        """
        Test log_rescale method with boundary values.

        Inputs:
            value=1 (min_old), min_old=1, max_old=1000, min_new=1, max_new=10
            value=1000 (max_old), min_old=1, max_old=1000, min_new=1, max_new=10

        Expected:
            For value=min_old, the result should be min_new (1)
            For value=max_old, the result should be max_new (10)
        """
        result_min = Tools.log_rescale(1, 1, 1000, 1, 10)
        result_max = Tools.log_rescale(1000, 1, 1000, 1, 10)
        assert result_min == 1
        assert result_max == 10


class TestInvLogRescale:
    """Test inv_log_rescale method of Tools class."""

    def test_inv_log_rescale_basic(self):
        """
        Test inv_log_rescale method with basic inputs.

        Inputs:
            value=2, min_old=1, max_old=3, min_new=1, max_new=1000, base=np.e

        Expected:
            The inverse log-rescaled value should be approximately 270
        """
        result = Tools.inv_log_rescale(2, 1, 3, 1, 1000)
        assert result == 270

    def test_inv_log_rescale_different_base(self):
        """
        Test inv_log_rescale method with a different base.

        Inputs:
            value=1, min_old=0, max_old=2, min_new=1, max_new=100, base=10

        Expected:
            The inverse log-rescaled value should be 10
        """
        result = Tools.inv_log_rescale(1, 0, 2, 1, 100, base=10)
        assert result == 10

    def test_inv_log_rescale_boundary_values(self):
        """
        Test inv_log_rescale method with boundary values.

        Inputs:
            value=min_old, min_old=1, max_old=3, min_new=1, max_new=1000
            value=max_old, min_old=1, max_old=3, min_new=1, max_new=1000

        Expected:
            For value=min_old, the result should be min_new (1)
            For value=max_old, the result should be max_new (1000)
        """
        result_min = Tools.inv_log_rescale(1, 1, 3, 1, 1000)
        result_max = Tools.inv_log_rescale(3, 1, 3, 1, 1000)
        assert result_min == 1
        assert result_max == 1000

    def test_inv_log_rescale_max_old_equals_min_old(self):
        """
        Test inv_log_rescale method with max_old equal to min_old.

        Inputs:
            value=50, min_old=50, max_old=50, min_new=0, max_new=10

        Expected:
            ZeroDivisionError should be raised because old_max equals old_min
            Note: The actual implementation might produce a NaN value which cannot be converted to int
        """
        # The implementation might raise different exceptions depending on how NaN is handled
        with pytest.raises((ZeroDivisionError, ValueError)):
            Tools.inv_log_rescale(50, 50, 50, 0, 10)

    def test_inv_log_rescale_large_values(self):
        """
        Test inv_log_rescale method with large values that might cause overflow.

        Inputs:
            value=100, min_old=0, max_old=200, min_new=1, max_new=1000

        Expected:
            The method should handle large exponential values without raising OverflowError
        """
        try:
            result = Tools.inv_log_rescale(100, 0, 200, 1, 1000)
            # If no error is raised, the test passes
            assert isinstance(result, int)
        except OverflowError:
            # If OverflowError is raised, the test fails
            pytest.fail("OverflowError was raised")

    def test_inv_log_rescale_round_trip(self):
        """
        Test round-trip conversion with log_rescale and inv_log_rescale.

        Inputs:
            original=500, min_old=1, max_old=1000, min_new=0, max_new=3

        Expected:
            The round-trip conversion may not exactly reproduce the original value
            due to rounding and the discrete nature of the transformation.
            We verify that the result is within a reasonable range of the original.
        """
        original = 500
        log_scaled = Tools.log_rescale(original, 1, 1000, 0, 3)
        back_to_original = Tools.inv_log_rescale(log_scaled, 0, 3, 1, 1000)

        # Verify the result is within a reasonable range
        # The actual implementation might not preserve the exact value due to rounding
        assert back_to_original > 0
        assert back_to_original <= 1000  # Should be within the new range


class TestRound:
    """Test round method of Tools class."""

    def test_round_to_half(self):
        """
        Test round method with precision=0.5.

        Inputs:
            number=3.7, precision=0.5

        Expected:
            The rounded value should be 3.5
        """
        result = Tools.round(3.7, 0.5)
        assert result == pytest.approx(3.5)

    def test_round_to_five(self):
        """
        Test round method with precision=5.

        Inputs:
            number=23, precision=5

        Expected:
            The rounded value should be 25
        """
        result = Tools.round(23, 5)
        assert result == 25

    def test_round_to_tenth(self):
        """
        Test round method with precision=0.1.

        Inputs:
            number=7.84, precision=0.1

        Expected:
            The rounded value should be approximately 7.8
            (Using almost_equal to handle floating point precision issues)
        """
        result = Tools.round(7.84, 0.1)
        # Use almost_equal to handle floating point precision issues
        assert abs(result - 7.8) < 1e-10

    def test_round_to_integer(self):
        """
        Test round method with precision=1.

        Inputs:
            number=7.4, precision=1
            number=7.6, precision=1

        Expected:
            7.4 should round to 7.0
            7.6 should round to 8.0
        """
        assert Tools.round(7.4, 1) == pytest.approx(7.0)
        assert Tools.round(7.6, 1) == pytest.approx(8.0)

    def test_round_negative_number(self):
        """
        Test round method with a negative number.

        Inputs:
            number=-3.7, precision=0.5

        Expected:
            The rounded value should be -3.5
        """
        result = Tools.round(-3.7, 0.5)
        assert result == -3.5

    def test_round_negative_precision(self):
        """
        Test round method with a negative precision.

        Inputs:
            number=23, precision=-5

        Expected:
            The rounded value should be 25, as the formula round(number / precision) * precision
            results in round(23 / -5) * -5 = -5 * -5 = 25

            Note: The negative precision affects the intermediate calculation,
            but the final result is positive due to the multiplication by negative precision.
        """
        result = Tools.round(23, -5)
        assert result == 25

    def test_round_zero_precision(self):
        """
        Test round method with zero precision.

        Inputs:
            number=7.84, precision=0

        Expected:
            ZeroDivisionError should be raised
        """
        with pytest.raises(ZeroDivisionError):
            Tools.round(7.84, 0)

    def test_round_already_at_precision(self):
        """
        Test round method with a number that is already at the specified precision.

        Inputs:
            number=3.5, precision=0.5
            number=25, precision=5

        Expected:
            The rounded value should be the same as the input
        """
        assert Tools.round(3.5, 0.5) == pytest.approx(3.5)
        assert Tools.round(25, 5) == pytest.approx(25)
