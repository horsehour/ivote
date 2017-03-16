package com.horsehour.vote.models.noise;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.DiagonalMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import com.google.common.collect.Multiset.Entry;
import com.google.common.primitives.Ints;
import com.horsehour.vote.Profile;
import com.horsehour.vote.stat.BiNormalGenzDist;
import com.horsehour.vote.stat.MultivariateNormal;
import com.horsehour.vote.stat.NormalDist;
import com.horsehour.vote.stat.MultivariateNormal.CDFResult;

public class NormalLogLikelihood {

	RealVector mean;
	RealVector variance;

	final ExecutorService exec;
	final MultivariateNormal mvn;

	public NormalLogLikelihood(RealVector mean, RealVector variance, ExecutorService exec, MultivariateNormal mvn) {
		this.mean = mean;
		this.variance = variance;
		this.exec = exec;
		this.mvn = mvn;
	}

	public NormalLogLikelihood(RealVector mean, RealVector variance, ExecutorService exec) {
		this(mean, variance, exec, MultivariateNormal.DEFAULT_INSTANCE);
	}

	/**
	 * Create a likelihood using the vectors as reference parameters.
	 * 
	 * @param mean
	 * @param variance
	 */
	public NormalLogLikelihood(RealVector mean, RealVector variance) {
		this(mean, variance, null, MultivariateNormal.DEFAULT_INSTANCE);
	}

	public <T> double logLikelihood(Profile<T> candidates, List<T> ordering) {
		List<int[]> indices = candidates.getIndices(ordering);
		return logLikelihood(indices);
	}

	/**
	 * Computes log likelihood. Unique-ifies identical rankings for efficiency.
	 * 
	 * @param indices
	 * @return
	 */
	public double logLikelihood(List<int[]> indices) {
		Multiset<List<Integer>> counts = HashMultiset.create();

		for (int[] ranking : indices)
			counts.add(Ints.asList(ranking));

		return logLikelihood(counts);
	}

	public double logLikelihood(Multiset<List<Integer>> counts) {
		double ll = 0;
		if (exec != null) {
			// Run in parallel
			List<LLResult> tasks = new ArrayList<LLResult>(counts.entrySet().size());

			for (Entry<List<Integer>> e : counts.entrySet())
				tasks.add(new LLResult(e.getCount(), Ints.toArray(e.getElement())));

			try {
				for (Future<Double> future : exec.invokeAll(tasks)) {
					ll += future.get();
				}
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
		} else {
			for (Entry<List<Integer>> e : counts.entrySet())
				ll += e.getCount() * singleRankingLL(Ints.toArray(e.getElement()));
		}
		return ll;
	}

	/**
	 * Non-parallelized, non-aggregated version of the likelihood
	 * 
	 * @param indices
	 * @return
	 */
	public double logLikelihoodNaive(List<int[]> indices) {
		double ll = 0;
		for (int[] ranking : indices)
			ll += singleRankingLL(ranking);
		return ll;
	}

	public class LLResult implements Callable<Double> {
		final int multiplicity;
		final int[] ranking;

		public LLResult(int multiplicity, int[] ranking) {
			this.multiplicity = multiplicity;
			this.ranking = ranking;
		}

		@Override
		public Double call() throws Exception {
			return multiplicity * singleRankingLL(ranking);
		}
	}

	double singleRankingLL(int[] ranking) {
		double value;

		if (ranking.length == 2)
			value = univariateLL(ranking);
		else if (ranking.length == 3)
			value = bivariateLL(ranking);
		else
			value = multivariateLL(ranking);

		return value;
	}

	/**
	 * Log likelihood computed using multivariate normal cdf
	 * 
	 * @param ranking
	 * @return
	 */
	double multivariateLL(int[] ranking) {
		return Math.log(multivariateProb(ranking).cdf);
	}

	public CDFResult multivariateProb(int[] ranking) {
		int n = ranking.length;
		RealMatrix d = new DiagonalMatrix(variance.toArray(), false);

		double[] lower = new double[n - 1];
		double[] upper = new double[n - 1];

		// Compute means and covariances of normal RVs representing differences
		RealMatrix a = new Array2DRowRealMatrix(n - 1, n);
		for (int i = 0; i < n - 1; i++) {
			a.setEntry(i, ranking[i] - 1, 1.0);
			a.setEntry(i, ranking[i + 1] - 1, -1.0);

			upper[i] = Double.POSITIVE_INFINITY;
		}

		RealVector mu = a.transpose().preMultiply(mean);
		RealMatrix sigma = a.multiply(d).multiply(a.transpose());

		return mvn.cdf(mu, sigma, lower, upper);
	}

	double bivariateLL(int[] ranking) {
		double m1 = mean.getEntry(ranking[0] - 1);
		double m2 = mean.getEntry(ranking[1] - 1);
		double m3 = mean.getEntry(ranking[2] - 1);
		double v1 = variance.getEntry(ranking[0] - 1);
		double v2 = variance.getEntry(ranking[1] - 1);
		double v3 = variance.getEntry(ranking[2] - 1);

		double mu1 = m2 - m1;
		double mu2 = m3 - m2;
		double sigma1 = Math.sqrt(v1 + v2);
		double sigma2 = Math.sqrt(v2 + v3);
		double rho = -v2 / (sigma1 * sigma2);

		double prob = BiNormalGenzDist.cdf(-mu1 / sigma1, -mu2 / sigma2, rho);
		return Math.log(prob);
	}

	double univariateLL(int[] ranking) {
		double m1 = mean.getEntry(ranking[0] - 1);
		double m2 = mean.getEntry(ranking[1] - 1);
		double v1 = variance.getEntry(ranking[0] - 1);
		double v2 = variance.getEntry(ranking[1] - 1);
		// prob of x_1 - x_2 > 0 or x_2 - x_1 < 0
		double prob = NormalDist.cdf01((m1 - m2) / Math.sqrt(v1 + v2)); // (0 -
																		// (m2 -
																		// m1))
																		// /
																		// sigma
		return Math.log(prob);
	}

	/**
	 * Incorrect log likelihood computed using bivariate normal
	 * 
	 * @param ranking
	 * @return
	 */
	double bogoLogLikelihood(int[] ranking) {
		// Special case when ranking is only 2 items long
		if (ranking.length == 2) {
			double m1 = mean.getEntry(ranking[0] - 1);
			double m2 = mean.getEntry(ranking[1] - 1);
			double v1 = variance.getEntry(ranking[0] - 1);
			double v2 = variance.getEntry(ranking[1] - 1);
			// prob of x_1 - x_2 > 0 or x_2 - x_1 < 0
			double prob = NormalDist.cdf01((m1 - m2) / Math.sqrt(v1 + v2)); // (0
																			// -
																			// (m2
																			// -
																			// m1))
																			// /
																			// sigma
			return Math.log(prob);
		}

		/*
		 * log likelihood as computed using
		 * http://math.stackexchange.com/questions/270745/compute-probability-of
		 * -a-particular-ordering-of-normal-random-variables
		 */
		double ll = 0;

		// add all bivariate triples
		for (int i = 0; i < ranking.length - 2; i++) {
			/*
			 * joint probability of (x_i > x_i+1, x_i+1 > x_i+2 ) equal to
			 * (x_i+1 - x_i < 0, x_i+2 - x_i+1 < 0)
			 */
			double m1 = mean.getEntry(ranking[i] - 1);
			double m2 = mean.getEntry(ranking[i + 1] - 1);
			double m3 = mean.getEntry(ranking[i + 2] - 1);
			double v1 = variance.getEntry(ranking[i] - 1);
			double v2 = variance.getEntry(ranking[i + 1] - 1);
			double v3 = variance.getEntry(ranking[i + 2] - 1);

			double mu1 = m2 - m1;
			double mu2 = m3 - m2;
			double sigma1 = Math.sqrt(v1 + v2);
			double sigma2 = Math.sqrt(v2 + v3);
			double rho = -v2 / (sigma1 * sigma2);

			double prob = BiNormalGenzDist.cdf(-mu1 / sigma1, -mu2 / sigma2, rho);
			ll += Math.log(prob);
		}

		// subtract all pairs that are not at the beginning of end
		for (int i = 1; i < ranking.length - 2; i++) {
			double m1 = mean.getEntry(ranking[i] - 1);
			double m2 = mean.getEntry(ranking[i + 1] - 1);
			double v1 = variance.getEntry(ranking[i] - 1);
			double v2 = variance.getEntry(ranking[i + 1] - 1);
			// prob of x_1 - x_2 > 0 or x_2 - x_1 < 0
			double prob = NormalDist.cdf01((m1 - m2) / Math.sqrt(v1 + v2)); // (0
																			// -
																			// (m2
																			// -
																			// m1))
																			// /
																			// sigma
			ll -= Math.log(prob);
		}

		return ll;
	}

}
