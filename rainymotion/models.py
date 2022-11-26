from utils import RYScaler
import numpy as np
import cv2
from ipol import Idw
import ipol
from scipy.ndimage import map_coordinates
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# interpolation routine
def _interpolator(points, coord_source, coord_target, method="idw"):

    coord_source_i, coord_source_j = coord_source
    coord_target_i, coord_target_j = coord_target

    # reshape
    trg = np.vstack((coord_source_i.ravel(), coord_source_j.ravel())).T
    src = np.vstack((coord_target_i.ravel(), coord_target_j.ravel())).T

    if method == "nearest":
        interpolator = NearestNDInterpolator(src, points.ravel(),
                                             tree_options={"balanced_tree": False})
        points_interpolated = interpolator(trg)
    elif method == "linear":
        interpolator = LinearNDInterpolator(src, points.ravel(), fill_value=0)
        points_interpolated = interpolator(trg)
    elif method == "idw":
        interpolator = ipol.Idw(src, trg)
        points_interpolated = interpolator(points.ravel())

    # reshape output
    points_interpolated = points_interpolated.reshape(points.shape)

    return points_interpolated.astype(points.dtype)


# constant-vector advection
def _advection_constant_vector(of_instance, lead_steps=12):

    delta_x = of_instance[::, ::, 0]
    delta_y = of_instance[::, ::, 1]

    # make a source meshgrid
    coord_source_i, coord_source_j = np.meshgrid(range(of_instance.shape[1]),
                                                 range(of_instance.shape[0]))

    # calculate new coordinates of radar pixels
    coord_targets = []
    for lead_step in range(lead_steps):
        coord_target_i = coord_source_i + delta_x * (lead_step + 1)
        coord_target_j = coord_source_j + delta_y * (lead_step + 1)
        coord_targets.append([coord_target_i, coord_target_j])

    coord_source = [coord_source_i, coord_source_j]

    return coord_source, coord_targets

# semi-Lagrangian advection
def _advection_semi_lagrangian(of_instance, lead_steps=12):

    delta_x = of_instance[::, ::, 0]
    delta_y = of_instance[::, ::, 1]

    # make a source meshgrid
    coord_source_i, coord_source_j = np.meshgrid(range(of_instance.shape[1]),
                                                 range(of_instance.shape[0]))

    # create dynamic delta holders
    delta_xi = delta_x.copy()
    delta_yi = delta_y.copy()

    # Block for calculation displacement
    # init placeholders
    coord_targets = []
    for lead_step in range(lead_steps):

        # calculate corresponding targets
        coord_target_i = coord_source_i + delta_xi
        coord_target_j = coord_source_j + delta_yi

        coord_targets.append([coord_target_i, coord_target_j])

        # now update source coordinates
        coord_source_i = coord_target_i
        coord_source_j = coord_target_j
        coord_source = [coord_source_j.ravel(), coord_source_i.ravel()]

        # update deltas
        delta_xi = map_coordinates(delta_x, coord_source).reshape(of_instance.shape[0], of_instance.shape[1])
        delta_yi = map_coordinates(delta_y, coord_source).reshape(of_instance.shape[0], of_instance.shape[1])

    # reinitialization of coordinates source
    coord_source_i, coord_source_j = np.meshgrid(range(of_instance.shape[1]),
                                                 range(of_instance.shape[0]))
    coord_source = [coord_source_i, coord_source_j]

    return coord_source, coord_targets

# calculate optical flow
def _calculate_of(data_instance,
                  method="DIS",
                  direction="forward"):

    # define frames order
    if direction == "forward":
        prev_frame = data_instance[-2]
        next_frame = data_instance[-1]
        coef = 1.0
    elif direction == "backward":
        prev_frame = data_instance[-1]
        next_frame = data_instance[-2]
        coef = -1.0

    # calculate dense flow
    if method == "Farneback":
        of_instance = cv2.optflow.createOptFlow_Farneback()
    elif method == "DIS":
        of_instance = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    elif method == "DeepFlow":
        of_instance = cv2.optflow.createOptFlow_DeepFlow()
    elif method == "PCAFlow":
        of_instance = cv2.optflow.createOptFlow_PCAFlow()
    elif method == "SimpleFlow":
        of_instance = cv2.optflow.createOptFlow_SimpleFlow()
    elif method == "SparseToDense":
        of_instance = cv2.optflow.createOptFlow_SparseToDense()

    print(f"prev_frame: {prev_frame.shape}")
    print(f"next_frame: {next_frame.shape}")

    # # 1 channel grayscale
    # prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # # 3 channel grayscale
    # prev_frame = np.stack((prev_frame,)*3, axis=-1)
    # next_frame = np.stack((next_frame,)*3, axis=-1)

    # prev_frame = prev_frame.reshape((120,120,1))
    # next_frame = next_frame.reshape((120,120,1))

    print(f"prev_frame: {prev_frame.shape}")
    print(f"next_frame: {next_frame.shape}")

    delta = of_instance.calc(prev_frame, next_frame, None) * coef
    print("got delta")

    if method in ["Farneback", "SimpleFlow"]:
        # variational refinement
        delta = cv2.optflow.createVariationalFlowRefinement().calc(prev_frame, next_frame, delta)
        delta = np.nan_to_num(delta)
        delta = _fill_holes(delta)

    return delta



def _fill_holes(of_instance, threshold=0):

    # calculate velocity scalar
    vlcty = np.sqrt(of_instance[::, ::, 0]**2 + of_instance[::, ::, 1]**2)

    # zero mask
    zero_holes = vlcty <= threshold

    # targets
    coord_target_i, coord_target_j = np.meshgrid(range(of_instance.shape[1]),
                                                 range(of_instance.shape[0]))

    # source
    coord_source_i, coord_source_j = coord_target_i[~zero_holes], coord_target_j[~zero_holes]
    delta_x_source = of_instance[::, ::, 0][~zero_holes]
    delta_y_source = of_instance[::, ::, 1][~zero_holes]

    # reshape
    src = np.vstack((coord_source_i.ravel(), coord_source_j.ravel())).T
    trg = np.vstack((coord_target_i.ravel(), coord_target_j.ravel())).T

    # create an object
    interpolator = ipol.Idw(src, trg)

    #
    delta_x_target = interpolator(delta_x_source.ravel())
    delta_y_target = interpolator(delta_y_source.ravel())

    # reshape output
    delta_x_target = delta_x_target.reshape(of_instance.shape[0],
                                            of_instance.shape[1])
    delta_y_target = delta_y_target.reshape(of_instance.shape[0],
                                            of_instance.shape[1])

    return np.stack([delta_x_target, delta_y_target], axis=-1)


class Dense:
    """
    The basic class for the Dense model of the rainymotion library.
    To run your nowcasting model you first have to set up a class instance
    as follows:
    `model = Dense()`
    and then use class attributes to set up model parameters, e.g.:
    `model.of_method = "DIS"`
    All class attributes have default values, for getting started with
    nowcasting you must specify only `input_data` attribute which holds the
    latest radar data observations. After specifying the input data, you can
    run nowcasting model and produce the corresponding results of nowcasting
    using `.run()` method:
    `nowcasts = model.run()`
    Attributes
    ----------
    input_data: 3D numpy array (frames, dim_x, dim_y) of radar data for
    previous hours. "frames" dimension must be > 2.
    scaler: function, default=rainymotion.utils.RYScaler
        Corner identification and optical flow algorithms require specific data
        type to perform calculations: uint8. That means that you must specify
        the transformation function (i.e. "scaler") to convert the "input_data"
        to the range of integers [0, 255]. By default we are using RYScaler
        which converts precipitation depth (mm, float16) to "brightness"
        values (uint8).
    lead_steps: int, default=12
        Number of lead times for which we want to produce nowcasts. Must be > 0
    of_method: str, default="DIS", options=["DIS", "PCAFlow", "DeepFlow",
                                            "Farneback"]
        The optical flow method to obtain the dense representation (in every
        image pixel) of motion field. By default we use the Dense Inverse
        Search algorithm (DIS). PCAFlow, DeepFlow, and Farneback algoritms
        are also available to obtain motion field.
    advection: str, default="constant-vector"
        The advection scheme we use for extrapolation of every image pixel
        into the imminent future.
    direction: str, default="backward", options=["forward", "backward"]
        The direction option of the advection scheme.
    interpolation: str, default="idw", options=["idw", "nearest", "linear"]
        The interpolation method we use to interpolate advected pixel values
        to the original grid of the radar image. By default we use inverse
        distance weightning interpolation (Idw) as proposed in library wradlib
        (wradlib.ipol.Idw), but interpolation techniques from scipy.interpolate
        (e.g., "nearest" or "linear") could also be used.
    Methods
    -------
    run(): perform calculation of nowcasts.
        Return 3D numpy array of shape (lead_steps, dim_x, dim_y).
    """

    def __init__(self):

        self.input_data = None

        self.scaler = RYScaler

        self.lead_steps = 12

        self.of_method = "DIS"

        self.direction = "backward"

        self.advection = "constant-vector"

        self.interpolation = "idw"

    def run(self):
        """
        Run nowcasting calculations.
        Returns
        -------
        nowcasts : 3D numpy array of shape (lead_steps, dim_x, dim_y).
        """

        # scale input data to uint8 [0-255] with self.scaler
        scaled_data, c1, c2 = self.scaler(self.input_data)

        # calculate optical flow
        of = _calculate_of(scaled_data, method=self.of_method,
                           direction=self.direction)

        # advect pixels accordingly
        if self.advection == "constant-vector":
            coord_source, coord_targets = _advection_constant_vector(of, lead_steps=self.lead_steps)
        elif self.advection == "semi-lagrangian":
            coord_source, coord_targets = _advection_semi_lagrangian(of, lead_steps=self.lead_steps)

        # nowcasts placeholder
        nowcasts = []

        # interpolation
        for lead_step in range(self.lead_steps):
            nowcasts.append(_interpolator(self.input_data[-1], coord_source,
                                          coord_targets[lead_step],
                                          method=self.interpolation))

        # reshaping
        nowcasts = np.moveaxis(np.dstack(nowcasts), -1, 0)

        return nowcasts