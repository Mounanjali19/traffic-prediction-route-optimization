import numpy as np
from datetime import datetime, timedelta


# ============================================================
# ROAD-SPECIFIC PARAMETERS
# ============================================================
#
# Each of the 30 selected road segments gets a fixed
# junction/road factor.
#
# This factor represents a road-specific characteristic
# that can influence the generated traffic conditions.

JUNCTION_FACTORS = np.random.uniform(
    0.8,
    1.3,
    size=30
)


# ============================================================
# TRAFFIC SEQUENCE GENERATION
# ============================================================

def generate_traffic_sequence(
    timestamp,
    scenario="normal"
):
    """
    Generate the traffic-history sequence used by
    the Hybrid GAT-LSTM model.

    Parameters
    ----------
    timestamp : str
        Prediction timestamp.

        Expected format:
            YYYY-MM-DDTHH:MM

    scenario : str
        Traffic scenario.

        Supported scenarios:
            normal
            rain
            event

    Returns
    -------
    numpy.ndarray

        Shape:

            (12, 30, 8)

        where:

            12 = historical time steps
            30 = road segments
             8 = features per road
    """

    # ========================================================
    # 1. CONVERT TIMESTAMP
    # ========================================================

    dt = datetime.fromisoformat(
        timestamp
    )


    # ========================================================
    # 2. CREATE 12 HISTORICAL TIME STEPS
    # ========================================================
    #
    # Traffic observations are spaced 5 minutes apart.
    #
    # 12 observations × 5 minutes
    # = approximately 1 hour of historical context.
    #
    # If the prediction time is 10:00, the sequence contains:
    #
    # 09:05
    # 09:10
    # 09:15
    # ...
    # 09:55
    # 10:00

    time_steps = [

        dt - timedelta(
            minutes=5 * i
        )

        for i in reversed(
            range(12)
        )

    ]


    # ========================================================
    # 3. STORAGE FOR THE COMPLETE SEQUENCE
    # ========================================================

    sequence = []


    # ========================================================
    # 4. GENERATE FEATURES FOR EVERY TIME STEP
    # ========================================================

    for current_time in time_steps:

        # ----------------------------------------------------
        # Extract temporal information
        # ----------------------------------------------------

        hour = current_time.hour

        minute = current_time.minute


        # Convert time into minutes from midnight.
        #
        # Example:
        #
        # 10:30
        # = 10 × 60 + 30
        # = 630

        minute_of_day = (
            hour * 60
            + minute
        )


        # Python weekday:
        #
        # Monday = 0
        # Sunday = 6

        day_of_week = (
            current_time.weekday()
        )


        # Weekend indicator:
        #
        # 0 → weekday
        # 1 → weekend

        is_weekend = int(
            day_of_week >= 5
        )


        # ----------------------------------------------------
        # Determine whether the current time is a peak period.
        #
        # Morning peak:
        #     07:00 - 10:00
        #
        # Evening peak:
        #     17:00 - 20:00
        # ----------------------------------------------------

        morning_peak = (
            7 <= hour < 10
        )

        evening_peak = (
            17 <= hour < 20
        )

        is_peak = (
            morning_peak
            or evening_peak
        )


        # ----------------------------------------------------
        # Base traffic speed
        # ----------------------------------------------------
        #
        # Peak periods reduce the base speed.
        #
        # This creates a controlled traffic pattern where
        # congestion is stronger during expected peak hours.

        if is_peak:

            base_speed = 22.0

        else:

            base_speed = 35.0


        # ----------------------------------------------------
        # Scenario effects
        # ----------------------------------------------------
        #
        # Rain and event scenarios further reduce the
        # expected traffic speed.

        if scenario == "rain":

            base_speed *= 0.75

        elif scenario == "event":

            base_speed *= 0.60


        # ----------------------------------------------------
        # Generate information for all 30 roads
        # ----------------------------------------------------

        road_features = []


        for road_index in range(30):

            # ------------------------------------------------
            # Road-specific speed variation
            # ------------------------------------------------
            #
            # Different roads can have slightly different
            # observed speeds around the base traffic condition.

            speed = (
                base_speed
                * JUNCTION_FACTORS[
                    road_index
                ]
                * np.random.uniform(
                    0.85,
                    1.15
                )
            )


            # ------------------------------------------------
            # Vehicle count
            # ------------------------------------------------
            #
            # Vehicle count is generated from the traffic
            # condition and road-specific factor.

            vehicle_count = int(
                max(
                    1,
                    100
                    * JUNCTION_FACTORS[
                        road_index
                    ]
                    * (
                        1.5
                        if is_peak
                        else 0.8
                    )
                    * np.random.uniform(
                        0.9,
                        1.1
                    )
                )
            )


            # ------------------------------------------------
            # Eight input features
            # ------------------------------------------------
            #
            # 1. Average speed
            # 2. Vehicle count
            # 3. Weekend indicator
            # 4. Rain indicator
            # 5. Event indicator
            # 6. Minute of day
            # 7. Day of week
            # 8. Junction factor

            features = [

                speed,

                vehicle_count,

                int(
                    is_weekend
                ),

                int(
                    scenario == "rain"
                ),

                int(
                    scenario == "event"
                ),

                minute_of_day,

                day_of_week,

                JUNCTION_FACTORS[
                    road_index
                ]

            ]


            road_features.append(
                features
            )


        # Add all 30 road observations
        # for the current time step.

        sequence.append(
            road_features
        )


    # ========================================================
    # 5. CONVERT TO NUMPY ARRAY
    # ========================================================

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )


    # ========================================================
    # 6. RETURN MODEL INPUT
    # ========================================================
    #
    # Final shape:
    #
    #     (12, 30, 8)
    #
    # 12 time steps
    # 30 roads
    # 8 features

    return sequence
