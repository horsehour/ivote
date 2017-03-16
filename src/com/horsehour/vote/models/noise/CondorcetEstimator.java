package com.horsehour.vote.models.noise;

import java.util.List;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.univariate.BrentOptimizer;
import org.apache.commons.math3.optim.univariate.SearchInterval;
import org.apache.commons.math3.optim.univariate.UnivariateObjectiveFunction;
import org.apache.commons.math3.optim.univariate.UnivariatePointValuePair;

import com.horsehour.vote.Profile;
import com.horsehour.vote.rule.KemenyYoung;

/**
 * Estimator for the Condorcet or Mallows model
 * @author mao
 *
 */
public class CondorcetEstimator implements OrdinalEstimator<CondorcetModel<?>> {

	BrentOptimizer brent;
	
	volatile double lastLL;
	
	public CondorcetEstimator() {
		brent = new BrentOptimizer(1e-7, 1e-11);
	}
	
	@Override
	public <T> CondorcetModel<T> fitModelOrdinal(final Profile<T> profile) {
		// Find optimal kemeny rankings
		KemenyYoung k = new KemenyYoung();			
		
		List<List<T>> bestRankings = k.getAllNearestRankings(profile);
//		System.out.println(bestRankings.size() + " rankings found");
		final List<T> someRanking = bestRankings.get(0);
		
//		double bestLL = Double.POSITIVE_INFINITY;
//		double bestPhi = 0;		
		
		/*
		 * Optimize p over each ranking and pick the one with the best likelihood
		 * RE: discussion with Hossein on 1/25: the likelihoods (and p) are all the same!
		 */
		UnivariateFunction logLk = new UnivariateFunction() {
			@Override public double value(double phi) {					
				return CondorcetModel.profileLogLikelihood(profile, someRanking, phi);
			}				
		};
		
		UnivariatePointValuePair result = 
				brent.optimize(
				new UnivariateObjectiveFunction(logLk),
				new SearchInterval(0, 1),
				new MaxEval(1000),
				GoalType.MAXIMIZE
				);
		
		double phi = result.getPoint();
		lastLL = result.getValue();
		
//		System.out.println("Model has p=" + 1/(1+phi) +", and likelihood " + ll);
		
//		for( final List<T> ranking : bestRankings ) {
//					
//			if( ll < bestLL ) {
//				bestLL = ll;
//				bestPhi = phi;
//				bestRanking = ranking;
//			}			
//		}
		
		CondorcetModel<T> cm = new CondorcetModel<T>(bestRankings, 1/(1+phi));
		cm.setFittedLikelihood(lastLL);
		return cm;
	}

	public String toString() {
		return this.getClass().getSimpleName();
	}
}
