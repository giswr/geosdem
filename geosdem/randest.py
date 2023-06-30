import numpy as np

def compute_dsdf(drought_data, severity_threshold, duration_thresholds):
    """Compute the Drought Severity Duration Frequency (DSDF) for a given set of drought data, 
        severity threshold, and duration thresholds.
    Args:
        drought_data (ndarray): a list or array of drought severity values
        severity_threshold (str): a float representing the severity threshold (e.g., 0.5 for moderate drought)
        duration_thresholds (ndarray): a list or array of duration thresholds in days (e.g., [7, 14, 30, 60])

    Returns:
        ndarray: A 2D numpy array containing the DSDF values for each duration threshold and return period
    """    
    # Sort the drought data in descending order
    sorted_data = np.sort(drought_data)[::-1]
    
    # Compute the number of years in the drought data
    num_years = len(drought_data) / 365
    
    # Initialize an empty 2D array to hold the DSDF values
    dsdf = np.empty((len(duration_thresholds), len(return_periods)))
    
    # Loop over each duration threshold
    for i, duration in enumerate(duration_thresholds):
        # Loop over each return period
        for j, period in enumerate(return_periods):
            # Compute the number of drought events exceeding the severity threshold for the given duration threshold
            num_events = len([x for x in sorted_data if x >= severity_threshold * duration])
            
            # Compute the probability of a drought event exceeding the severity threshold for the given duration threshold
            prob_event = num_events / num_years
            
            # Compute the DSDF value for the given duration threshold and return period
            dsdf[i, j] = (1 - prob_event) ** period
            
    return dsdf


from pandas._typing import CorrelationMethod
from pandas.core.interchange.dataframe_protocol import DataFrame
from pandas._libs import algos
#import sklearn.metrics as metrics
import hydroeval as he
def corr(
        self,
        method: CorrelationMethod = "pearson",
        min_periods: int = 1,
        numeric_only: bool = False,
    ) -> DataFrame:
        """
        Compute pairwise correlation of columns, excluding NA/null values.

        Parameters
        ----------
        method : {'pearson', 'kendall', 'spearman', 'nse'} or callable
            Method of correlation:

            * pearson : standard correlation coefficient
            * kendall : Kendall Tau correlation coefficient
            * spearman : Spearman rank correlation
            * callable: callable with input two 1d ndarrays
                and returning a float. Note that the returned matrix from corr
                will have 1 along the diagonals and will be symmetric
                regardless of the callable's behavior.
            * nse : nastche succlife efficiency
        min_periods : int, optional
            Minimum number of observations required per pair of columns
            to have a valid result. Currently only available for Pearson
            and Spearman correlation.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0
                The default value of ``numeric_only`` is now ``False``.

        Returns
        -------
        DataFrame
            Correlation matrix.

        See Also
        --------
        DataFrame.corrwith : Compute pairwise correlation with another
            DataFrame or Series.
        Series.corr : Compute the correlation between two Series.

        Notes
        -----
        Pearson, Kendall and Spearman correlation are currently computed using pairwise complete observations.

        * `Pearson correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_
        * `Kendall rank correlation coefficient <https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient>`_
        * `Spearman's rank correlation coefficient <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_

        Examples
        --------
        >>> def histogram_intersection(a, b):
        ...     v = np.minimum(a, b).sum().round(decimals=1)
        ...     return v
        >>> df = pd.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'])
        >>> df.corr(method=histogram_intersection)
              dogs  cats
        dogs   1.0   0.3
        cats   0.3   1.0

        >>> df = pd.DataFrame([(1, 1), (2, np.nan), (np.nan, 3), (4, 4)],
        ...                   columns=['dogs', 'cats'])
        >>> df.corr(min_periods=3)
              dogs  cats
        dogs   1.0   NaN
        cats   NaN   1.0
        """  # noqa: E501
        data = self._get_numeric_data() if numeric_only else self
        cols = data.columns
        idx = cols.copy()
        mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)

        if method == "nse":
            correl = he.evaluator(he.nse, evaluation=mat, simulations=mat)
        elif method == "pearson":
            correl = algos.nancorr(mat, minp=min_periods)
        elif method == "spearman":
            correl = algos.nancorr_spearman(mat, minp=min_periods)
        elif method == "kendall" or callable(method):
            if min_periods is None:
                min_periods = 1
            mat = mat.T
            corrf = nanops.get_corr_func(method)
            K = len(cols)
            correl = np.empty((K, K), dtype=float)
            mask = np.isfinite(mat)
            for i, ac in enumerate(mat):
                for j, bc in enumerate(mat):
                    if i > j:
                        continue

                    valid = mask[i] & mask[j]
                    if valid.sum() < min_periods:
                        c = np.nan
                    elif i == j:
                        c = 1.0
                    elif not valid.all():
                        c = corrf(ac[valid], bc[valid])
                    else:
                        c = corrf(ac, bc)
                    correl[i, j] = c
                    correl[j, i] = c
        else:
            raise ValueError(
                "method must be either 'pearson', "
                "'spearman', 'kendall', or a callable, "
                f"'{method}' was supplied"
            )

        result = self._constructor(correl, index=idx, columns=cols, copy=False)
        return result.__finalize__(self, method="corr")

def cov(
    self,
    min_periods: int | None = None,
    ddof: int | None = 1,
    numeric_only: bool = False,
) -> DataFrame:
    """
    Compute pairwise covariance of columns, excluding NA/null values.

    Compute the pairwise covariance among the series of a DataFrame.
    The returned data frame is the `covariance matrix
    <https://en.wikipedia.org/wiki/Covariance_matrix>`__ of the columns
    of the DataFrame.

    Both NA and null values are automatically excluded from the
    calculation. (See the note below about bias from missing values.)
    A threshold can be set for the minimum number of
    observations for each value created. Comparisons with observations
    below this threshold will be returned as ``NaN``.

    This method is generally used for the analysis of time series data to
    understand the relationship between different measures
    across time.

    Parameters
    ----------
    min_periods : int, optional
        Minimum number of observations required per pair of columns
        to have a valid result.

    ddof : int, default 1
        Delta degrees of freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        This argument is applicable only when no ``nan`` is in the dataframe.

    numeric_only : bool, default False
        Include only `float`, `int` or `boolean` data.

        .. versionadded:: 1.5.0

        .. versionchanged:: 2.0.0
            The default value of ``numeric_only`` is now ``False``.

    Returns
    -------
    DataFrame
        The covariance matrix of the series of the DataFrame.

    See Also
    --------
    Series.cov : Compute covariance with another Series.
    core.window.ewm.ExponentialMovingWindow.cov : Exponential weighted sample
        covariance.
    core.window.expanding.Expanding.cov : Expanding sample covariance.
    core.window.rolling.Rolling.cov : Rolling sample covariance.

    Notes
    -----
    Returns the covariance matrix of the DataFrame's time series.
    The covariance is normalized by N-ddof.

    For DataFrames that have Series that are missing data (assuming that
    data is `missing at random
    <https://en.wikipedia.org/wiki/Missing_data#Missing_at_random>`__)
    the returned covariance matrix will be an unbiased estimate
    of the variance and covariance between the member Series.

    However, for many applications this estimate may not be acceptable
    because the estimate covariance matrix is not guaranteed to be positive
    semi-definite. This could lead to estimate correlations having
    absolute values which are greater than one, and/or a non-invertible
    covariance matrix. See `Estimation of covariance matrices
    <https://en.wikipedia.org/w/index.php?title=Estimation_of_covariance_
    matrices>`__ for more details.

    Examples
    --------
    >>> df = pd.DataFrame([(1, 2), (0, 3), (2, 0), (1, 1)],
    ...                   columns=['dogs', 'cats'])
    >>> df.cov()
                dogs      cats
    dogs  0.666667 -1.000000
    cats -1.000000  1.666667

    >>> np.random.seed(42)
    >>> df = pd.DataFrame(np.random.randn(1000, 5),
    ...                   columns=['a', 'b', 'c', 'd', 'e'])
    >>> df.cov()
                a         b         c         d         e
    a  0.998438 -0.020161  0.059277 -0.008943  0.014144
    b -0.020161  1.059352 -0.008543 -0.024738  0.009826
    c  0.059277 -0.008543  1.010670 -0.001486 -0.000271
    d -0.008943 -0.024738 -0.001486  0.921297 -0.013692
    e  0.014144  0.009826 -0.000271 -0.013692  0.977795

    **Minimum number of periods**

    This method also supports an optional ``min_periods`` keyword
    that specifies the required minimum number of non-NA observations for
    each column pair in order to have a valid result:

    >>> np.random.seed(42)
    >>> df = pd.DataFrame(np.random.randn(20, 3),
    ...                   columns=['a', 'b', 'c'])
    >>> df.loc[df.index[:5], 'a'] = np.nan
    >>> df.loc[df.index[5:10], 'b'] = np.nan
    >>> df.cov(min_periods=12)
                a         b         c
    a  0.316741       NaN -0.150812
    b       NaN  1.248003  0.191417
    c -0.150812  0.191417  0.895202
    """
    data = self._get_numeric_data() if numeric_only else self
    cols = data.columns
    idx = cols.copy()
    mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)

    if notna(mat).all():
        if min_periods is not None and min_periods > len(mat):
            base_cov = np.empty((mat.shape[1], mat.shape[1]))
            base_cov.fill(np.nan)
        else:
            base_cov = np.cov(mat.T, ddof=ddof)
        base_cov = base_cov.reshape((len(cols), len(cols)))
    else:
        base_cov = algos.nancorr(mat, cov=True, minp=min_periods)

    result = self._constructor(base_cov, index=idx, columns=cols, copy=False)
    return result.__finalize__(self, method="cov")