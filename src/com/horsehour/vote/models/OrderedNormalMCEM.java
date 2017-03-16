package com.horsehour.vote.models;

import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.analysis.function.Abs;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import com.google.common.collect.Multiset.Entry;
import com.google.common.primitives.Ints;
import com.horsehour.vote.Profile;
import com.horsehour.vote.models.noise.MeanVarParams;
import com.horsehour.vote.models.noise.NormalLogLikelihood;
import com.horsehour.vote.models.noise.NormalNoiseModel;
import com.horsehour.vote.stat.MultivariateMean;
import com.horsehour.vote.stat.MultivariateNormal;

/**
 * This is a general implementation of the probit model as described at
 * https://wiki.ece.cmu.edu/ddl/index.php/Introduction_to_random_utility_discrete_choice_models
 * 
 * FIXME: Supposed to support independent variances, but currently broken
 * 
 * @author mao
 *
 * @param <T>
 */
public class OrderedNormalMCEM extends MCEMModel<NormalMoments, NormalNoiseModel<?>, MeanVarParams> {
	
	final boolean floatVariance;
	final int startingSamples;
	final int incrSamples;	
	
	MultivariateMean m1Stats;
	MultivariateMean m2Stats;
	
	RealVector delta, variance;
	NormalLogLikelihood ll;	
	volatile double lastLL;
	
	List<int[]> rankings;
	int numItems;	
	Multiset<List<Integer>> counts;
	
	/**
	 * Created an ordered normal model using MCEM. A fixed variance is set to 1.
	 * 
	 * @param floatVariance whether the variance should be allowed to change during EM.
	 */
	public OrderedNormalMCEM(boolean floatVariance, int maxIters, double abseps, double releps, int startingSamples, int incrSamples) {
		super(maxIters, abseps, releps);		
		this.floatVariance = floatVariance;		
		this.startingSamples = startingSamples;
		this.incrSamples = incrSamples;		
	}
	
	/**
	 * Default ordered normal MCEM model with 2000 starting samples and 300 add'l per iteration.
	 * 
	 * @param floatVariance
	 * @param maxIters
	 * @param abseps
	 * @param releps
	 */
	public OrderedNormalMCEM(boolean floatVariance, int maxIters, double abseps, double releps) {
		this(floatVariance, maxIters, abseps, releps, 2000, 300);
	}
	
	@Override
	protected void initialize(List<int[]> rankings, int m) {
		this.rankings = rankings;
		this.numItems = m;
		
		m1Stats = new MultivariateMean(m);				
		delta = new ArrayRealVector(start);
		
		if( floatVariance ) {
			m2Stats = new MultivariateMean(m);
			
			double[] randomVars = new NormalDistribution().sample(m);
			variance = new ArrayRealVector(randomVars).mapToSelf(new Abs()).mapAddToSelf(1);	
		}
		else {
			variance = new ArrayRealVector(m, 1.0d);
		}		
				
		ll = new NormalLogLikelihood(delta, variance, 
				EstimatorUtils.threadPool, MultivariateNormal.DEFAULT_INSTANCE);
				
		counts = HashMultiset.create();			
		for( int[] ranking : rankings )
			counts.add(Ints.asList(ranking));
	}

	@Override
	protected void eStep(int i) {
		/*
		 * E-step: parallelized Gibbs sampling
		 * # Samples are increased as we get closer to true goal			
		 */	
		int samples = startingSamples + incrSamples*i;
				
		m1Stats.clear();
		if( floatVariance ) m2Stats.clear();	
		
		for( Entry<List<Integer>> e : counts.entrySet() ) {
			int[] ranking = Ints.toArray(e.getElement());
			int weight = e.getCount();
			
			super.addJob(new NormalGibbsSampler(delta, variance, ranking, samples, floatVariance, weight));							
		}
	}

	@Override
	protected void addData(NormalMoments data) {
		for( int i = 0; i < data.weight; i++ ) {
			m1Stats.addValue(data.m1);
			if( floatVariance ) m2Stats.addValue(data.m2);	
		}
	}
	
	@Override
	protected void mStep() {
		/*
		 * M-step: re-compute parameters
		 */
		double[] eM1 = m1Stats.getMean();
		double[] eM2 = floatVariance ? m2Stats.getMean() : null;
		
		for( int i = 0; i < eM1.length; i++ ) {
			double m = eM1[i];
			delta.setEntry(i, m);			
			
			if( floatVariance ) variance.setEntry(i, eM2[i] - m*m);
		}
					
		/* 
		 * adjust the mean and variance values to prevent drift:
		 * first subtract means so that first value is 0
		 * then scale variance to 1
		 */	
		
		// Adjust all variables so that first var is 1 		 
		if( floatVariance ) {
			double var = variance.getEntry(0);			
			variance.mapDivideToSelf(var);
			delta.mapDivideToSelf(Math.sqrt(var));
		}
		
		// Re-center means - first mean is 0
		delta.mapSubtractToSelf(delta.getEntry(0));
		
		logger.debug("Mean: {}", delta);
		logger.debug("Variance: {}", variance);			
	}	

	@Override
	protected double getLogLikelihood() {		
		// Don't modify any parameters as this can happen multi-threaded
		return lastLL = ll.logLikelihood(counts);
	}	

	@Override
	protected MeanVarParams getFinalParameters() {
		// Return the LL that was just computed above
		return new MeanVarParams(delta.toArray(), variance.toArray(), lastLL);
	}

	@Override
	public <T> NormalNoiseModel<T> fitModelOrdinal(Profile<T> profile) {		
		List<T> ordering = Arrays.asList(profile.getSortedItems());
		List<int[]> rankings = profile.getIndices(ordering);
		
		// Default initialization if setup not called
		if (this.start == null || this.start.length != ordering.size() )
			setup(new NormalDistribution().sample(ordering.size()));
		
		MeanVarParams params = getParameters(rankings, ordering.size());		
		
		NormalNoiseModel<T> nn = new NormalNoiseModel<T>(ordering, params);
		nn.setFittedLikelihood(lastLL);
		
		this.start = null; // reset the start point for next run 
		
		return nn;		
	}

}
