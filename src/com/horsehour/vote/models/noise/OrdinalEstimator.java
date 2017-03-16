package com.horsehour.vote.models.noise;

import com.horsehour.vote.Profile;

public interface OrdinalEstimator<M extends NoiseModel<?>> {

	<T> M fitModelOrdinal(Profile<T> profile);

}
