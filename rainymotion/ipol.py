import numpy as np


from scipy import ndimage, spatial, special, stats
import scipy

import warnings

from packaging.version import Version


class MissingSourcesError(Exception):
    """Is raised in case no source coordinates are available for interpolation."""

class MissingTargetsError(Exception):
    """Is raised in case no interpolation targets are available."""


class IpolBase:
    """
    IpolBase(src, trg)
    The base class for interpolation in N dimensions.
    Provides the basic interface for all other classes.
    Parameters
    ----------
    src : :class:`numpy:numpy.ndarray`
        ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the source points.
    trg : :class:`numpy:numpy.ndarray`
        ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the target points.
    """

    def __init__(self, src, trg, **kwargs):
        src = self._make_coord_arrays(src)
        trg = self._make_coord_arrays(trg)
        self.numsources = len(src)
        self.numtargets = len(trg)

    def __call__(self, vals):
        """
        Evaluate interpolator for values given at the source points.
        Parameters
        ----------
        vals : :class:`numpy:numpy.ndarray`
            ndarray of float, shape (numsources, ...)
            Values at the source points which to interpolate
        Returns
        -------
        output : None
        """
        self._check_shape(vals)
        return None

    def _check_shape(self, vals):
        """
        Checks whether the values correspond to the source points
        Parameters
        ----------
        vals : :class:`numpy:numpy.ndarray`
            ndarray of float
        """
        assert len(vals) == self.numsources, (
            f"Length of value array {len(vals)} does not correspond to number "
            f"of source points {self.numsources}"
        )
        self.valsshape = vals.shape
        self.valsndim = vals.ndim

    def _make_coord_arrays(self, x):
        """
        Make sure that the coordinates are provided as ndarray
        of shape (numpoints, ndim)
        Parameters
        ----------
        x : :class:`numpy:numpy.ndarray`
            ndarray of float with shape (numpoints, ndim)
            OR a sequence of ndarrays of float with len(sequence)==ndim and
            the length of the ndarray corresponding to the number of points
        """
        if type(x) in [list, tuple]:
            x = [item.ravel() for item in x]
            x = np.array(x).transpose()
        elif type(x) == np.ndarray:
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            elif x.ndim == 2:
                pass
            else:
                raise Exception("Cannot deal wih 3-d arrays, yet.")
        return x

    def _make_2d(self, vals):
        """Reshape increase number of dimensions of vals if smaller than 2,
        appending additional dimensions (as opposed to the atleast_nd methods
        of numpy).
        Parameters
        ----------
        vals : :class:`numpy:numpy.ndarray`
            values who are to be reshaped to the right shape
        Returns
        -------
        output : :class:`numpy:numpy.ndarray`
            if vals.shape==() [a scalar] output.shape will be (1,1)
            if vals.shape==(npt,) output.shape will be (npt,1)
            if vals.ndim > 1 vals will be returned as is
        """
        if vals.ndim < 2:
            # ndmin might be 0 so we get it to 1-d first
            # then we add an axis as we assume that
            return np.atleast_1d(vals)[:, np.newaxis]
        else:
            return vals




class Idw(IpolBase):
    """
    Idw(src, trg, nnearest=4, p=2.)
    Inverse distance weighting interpolation in N dimensions.
    Parameters
    ----------
    src : :class:`numpy:numpy.ndarray`
        ndarray of floats, shape (npoints, ndims) of cKDTree object
        Data point coordinates of the source points.
    trg : :class:`numpy:numpy.ndarray`
        ndarray of floats, shape (npoints, ndims)
        Data point coordinates of the target points.
    nnearest : int
        max. number of neighbours to be considered
    p : float
        inverse distance power used in 1/dist**p
    remove_missing : bool
        If True masks NaN values in the data values, defaults to False
    Keyword Arguments
    -----------------
    **kwargs : dict
        keyword arguments of ipclass (see class documentation)
    Examples
    --------
    See :ref:`/notebooks/interpolation/wradlib_ipol_example.ipynb`.
    Note
    ----
    Uses :class:`scipy:scipy.spatial.cKDTree`
    """

    def __init__(self, src, trg, nnearest=4, p=2.0, remove_missing=False, **kwargs):

        if isinstance(src, spatial.cKDTree):
            self.tree = src
        else:
            src = self._make_coord_arrays(src)
            if len(src) == 0:
                raise MissingSourcesError
            # plant a tree, use unbalanced tree as default
            kwargs.update(balanced_tree=kwargs.pop("balanced_tree", False))
            self.tree = spatial.cKDTree(src, **kwargs)

        self.numsources = self.tree.n

        trg = self._make_coord_arrays(trg)
        self.numtargets = len(trg)
        if self.numtargets == 0:
            raise MissingTargetsError

        if nnearest > self.numsources:
            warnings.warn(
                "wradlib.ipol.Idw: <nnearest> is larger than number of "
                f"source points and is set to {self.numsources} corresponding to the "
                "number of source points.",
                UserWarning,
            )
            self.nnearest = self.numsources
        else:
            self.nnearest = nnearest

        self.remove_missing = remove_missing

        self.p = p
        # query tree
        # scipy kwarg changed from version 1.6
        if Version(scipy.__version__) < Version("1.6"):
            query_kwargs = dict(n_jobs=-1)
        else:
            query_kwargs = dict(workers=-1)
        self.dists, self.ix = self.tree.query(trg, k=self.nnearest, **query_kwargs)
        # avoid bug, if there is only one neighbor at all
        if self.dists.ndim == 1:
            self.dists = self.dists[:, np.newaxis]
            self.ix = self.ix[:, np.newaxis]

    def __call__(self, vals, maxdist=None):
        """
        Evaluate interpolator for values given at the source points.
        You can interpolate multiple datasets of source values (``vals``) at
        once: the ``vals`` array should have the shape (number of source
        points, number of source datasets). If you want to interpolate only one
        set of source values, ``vals`` can have the shape (number of source
        points, 1) or just (number of source points,) - which is a flat/1-D
        array. The output will have the same number of dimensions as ``vals``,
        i.e. it will be a flat 1-D array in case ``vals`` is a 1-D array.
        Parameters
        ----------
        vals : :class:`numpy:numpy.ndarray`
            ndarray of float, shape (numsourcepoints, ...)
            Values at the source points which to interpolate
        maxdist : float
            the maximum distance up to which points will be included into the
            interpolation calculation
        Returns
        -------
        output : :class:`numpy:numpy.ndarray`
            ndarray of float with shape (numtargetpoints,...)
        """
        self._check_shape(vals)
        # print(f"self.dists: {self.dists}")
        # print(f"self.p: {self.p}")
        weights = 1.0 / self.dists**self.p

        # if maxdist isn't given, take the maximum distance
        if maxdist is not None:
            outside = self.dists > maxdist
            weights[outside] = 0

        # take care of point coincidence
        weights[np.isposinf(weights)] = 1e12

        # shape handling (time, ensemble etc)
        wshape = weights.shape
        weights.shape = wshape + ((vals.ndim - 1) * (1,))

        # expand vals to trg grid
        trgvals = vals[self.ix]

        # nan handling
        if self.remove_missing:
            isnan = np.isnan(trgvals)
            weights = np.broadcast_to(weights, isnan.shape)
            masked_weights = np.ma.array(weights, mask=isnan)

            interpol = np.nansum(weights * trgvals, axis=1) / np.sum(
                masked_weights, axis=1
            )
        else:
            interpol = np.sum(weights * trgvals, axis=1) / np.sum(weights, axis=1)

        return interpol
