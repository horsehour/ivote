package com.horsehour.vote.models;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.atomic.AtomicInteger;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.horsehour.vote.models.noise.NoiseModel;

/**
 * Multithread Implementation of Random Utility Model via MC-EM
 *
 * @author Andrew Mao
 * @see Azari, Hossein, David Parks, and Lirong Xia.
 *      "Random Utility Theory for Social Choice." Advances in Neural
 *      Information Processing Systems. 2012.
 */
public abstract class MCEMModel<T, M extends NoiseModel<?>, P> extends RandomUtilityEstimator<M, P> {
	protected final Logger logger = LoggerFactory.getLogger(this.getClass());

	int maxIters;
	double abseps;
	double releps;
	double[] start;

	AtomicInteger submittedJobs;
	CompletionService<T> ecs;

	public MCEMModel(int maxIters, double abseps, double releps) {
		this.maxIters = maxIters;
		this.abseps = abseps;
		this.releps = releps;
	}

	public void setup(double[] startPoint) {
		this.start = startPoint;
	}

	/*
	 * Implemented by subclasses
	 */
	protected abstract void initialize(List<int[]> rankings, int numItems);

	protected abstract void eStep(int iter);

	protected abstract void addData(T data);

	protected abstract void mStep();

	protected abstract double getLogLikelihood();

	protected abstract P getFinalParameters();

	@Override
	public synchronized P getParameters(List<int[]> rankings, int numItems) {
		/*
		 * NOT reentrant. Don't call this from multiple threads.
		 */

		ecs = new ExecutorCompletionService<T>(EstimatorUtils.threadPool);
		submittedJobs = new AtomicInteger(0);

		initialize(rankings, numItems);
		double ll = Double.NEGATIVE_INFINITY;

		for (int i = 0; i < maxIters; i++) {
			submittedJobs.set(0);

			logger.debug("Starting iteration {}", i);

			eStep(i);

			// Wait for sampling to finish
			int jobs = submittedJobs.get();

			for (int j = 0; j < jobs; j++) {
				try {
					addData(ecs.take().get());
				} catch (InterruptedException e) {
					e.printStackTrace();
					j--;
				} catch (ExecutionException e) {
					e.getCause().printStackTrace();
				}
			}

			mStep();

			double newLL = getLogLikelihood();
			logger.debug("Likelihood: {}", newLL);
			double absImpr = newLL - ll;
			double relImpr = -absImpr / ll;

			if (absImpr < abseps) {
				// System.out.printf("Absolute tolerance reached: %f < %f\n",
				// absImpr, abseps);
				break;
			}
			if (!Double.isNaN(relImpr) && relImpr < releps) {
				// System.out.printf("Relative tolerance reached: %f < %f\n",
				// relImpr, releps);
				break;
			}

			ll = newLL;
		}

		return getFinalParameters();
	}

	void addJob(Callable<T> job) {
		submittedJobs.incrementAndGet();
		ecs.submit(job);
	}

}
