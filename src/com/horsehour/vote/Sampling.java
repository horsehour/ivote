package com.horsehour.vote;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import org.apache.commons.math3.distribution.UniformRealDistribution;

import com.horsehour.util.MathLib;

public class Sampling {

	static Random rand = new Random();

	/**
	 * Returns a subset of size k chosen at random from elements in [0, n)
	 * 
	 * @param n
	 * @param k
	 * @return
	 */
	public static int[] randNK(int n, int k, Random rnd) {
		if (rnd == null)
			rnd = Sampling.rand;

		// Fix bad inputs
		if (k > n)
			k = n;
		else if (k < 0)
			k = 0;

		int i, sw, temp;
		int[] arr = new int[n];

		for (i = 0; i < n; i++)
			arr[i] = i;

		// Randomly arrange the first k elements
		for (i = 0; i < k; i++) {
			sw = i + rnd.nextInt(n - i);

			// Swap
			temp = arr[sw];
			arr[sw] = arr[i];
			arr[i] = temp;
		}

		return Arrays.copyOf(arr, k);
	}

	/**
	 * The random stream selection algorithm
	 * 
	 * @param <T>
	 * @param iter
	 * @return
	 */
	public static <T> T selectRandom(Iterable<T> iter) {
		int count = 0;
		T selected = null;

		for (T current : iter) {
			count += 1;

			if (rand.nextDouble() < 1.0 / count) {
				selected = current;
			}
		}
		return selected;
	}

	/**
	 * Selects indices from an array based on weights.
	 * 
	 * @param wts
	 * @param rnd
	 * @return
	 */
	public static int selectRandomWeighted(double[] wts, Random rnd) {
		int selected = 0;
		double total = wts[0];

		for (int i = 1; i < wts.length; i++) {
			total += wts[i];

			if (rnd.nextDouble() <= (wts[i] / total)) {
				selected = i;
			}
		}

		return selected;
	}

	/**
	 * N choose K from a list.
	 * 
	 * @param <T>
	 * @param list
	 * @param k
	 * @return
	 */
	public static <T> Collection<T> selectKRandom(List<T> list, int k) {
		int[] indices = randNK(list.size(), k, rand);

		Collection<T> stuff = new LinkedList<T>();
		for (int i = 0; i < indices.length; i++) {
			stuff.add(list.get(indices[i]));
		}

		return stuff;
	}

	/**
	 * N choose K from a list.
	 * 
	 * @param <T>
	 * @param list
	 * @param k
	 * @param rnd
	 * @return
	 */
	public static <T> Collection<T> selectKRandom(List<T> list, int k, Random rnd) {
		if (rnd == null)
			rnd = Sampling.rand;

		int[] indices = randNK(list.size(), k, rnd);

		Collection<T> stuff = new LinkedList<T>();
		for (int i = 0; i < indices.length; i++) {
			stuff.add(list.get(indices[i]));
		}

		return stuff;
	}

	/**
	 * N choose K from an array.
	 * 
	 * @param <T>
	 * @param arr
	 * @param k
	 * @return
	 */
	public static <T> T[] selectKRandom(T[] arr, int k, Random rnd) {
		int[] indices = randNK(arr.length, k, rnd);

		T[] stuff = Arrays.copyOf(arr, k);

		for (int i = 0; i < indices.length; i++) {
			stuff[i] = arr[indices[i]];
		}

		return stuff;
	}

	/**
	 * Randomly shuffles an array of integers in place
	 * 
	 * @param arr
	 * @param rnd
	 * @return
	 */
	public static int[] shuffle(int[] arr, Random rnd) {
		if (rnd == null)
			rnd = Sampling.rand;

		int temp;

		for (int i = arr.length - 1; i > 0; i--) {
			int j = rnd.nextInt(i + 1);

			temp = arr[j];
			arr[j] = arr[i];
			arr[i] = temp;
		}

		return arr;
	}

	public static long[] shuffle(long[] arr, Random rnd) {
		if (rnd == null)
			rnd = Sampling.rand;

		long temp;

		for (int i = arr.length - 1; i > 0; i--) {
			int j = rnd.nextInt(i + 1);

			temp = arr[j];
			arr[j] = arr[i];
			arr[i] = temp;
		}

		return arr;
	}

	/**
	 * Randomly shuffles an array in place
	 * 
	 * @param <T>
	 * @param arr
	 * @param rnd
	 * @return
	 */
	public static <T> T[] shuffle(T[] arr, Random rnd) {
		if (rnd == null)
			rnd = Sampling.rand;

		T temp;

		for (int i = arr.length - 1; i > 0; i--) {
			int j = rnd.nextInt(i + 1);

			temp = arr[j];
			arr[j] = arr[i];
			arr[i] = temp;
		}

		return arr;
	}

	/**
	 * Generates a random permutation of numbers from 0 to size - 1
	 * 
	 * @param size
	 * @return
	 */
	public static int[] randomShuffle(int size) {
		int[] arr = new int[size];

		for (int i = 0; i < size; i++)
			arr[i] = i;
		Sampling.shuffle(arr, null);

		return arr;
	}

	/**
	 * Generates a random permutation of numbers from 0 to size - 1
	 * 
	 * @param size
	 * @param rnd
	 * @return
	 */
	public static int[] randomShuffle(int size, Random rnd) {
		int[] arr = new int[size];

		for (int i = 0; i < size; i++)
			arr[i] = i;
		Sampling.shuffle(arr, rnd);

		return arr;
	}

	/**
	 * Generates a random shuffle of a list
	 * 
	 * @param <T>
	 * @param stuff
	 * @return
	 */
	@SuppressWarnings("unchecked")
	public static <T> List<T> randomShuffle(List<T> stuff) {
		T[] arr = (T[]) stuff.toArray();
		Sampling.shuffle(arr, null);
		return Arrays.asList(arr);
	}

	/**
	 * Generates an array of n items, each independently from 0 to c-1, which
	 * are always the same with the same seed
	 * 
	 * @param n
	 * @param c
	 * @param seed
	 * @return
	 */
	public static int[] randomSeededMultiset(int n, int c, long seed) {
		int[] arr = new int[n];
		Random seededRand = new Random(seed);
		for (int i = 0; i < n; i++)
			arr[i] = seededRand.nextInt(c);
		return arr;
	}

	/**
	 * Sampling with replacement
	 * 
	 * @param numSample
	 * @param b
	 * @return Sample of size numSample from [0, b)
	 */
	public static List<Integer> sampleMultiset(int numSample, int b) {
		List<Integer> result = new ArrayList<>();
		long seed = System.currentTimeMillis();
		Random rnd = new Random(seed);
		for (int i = 0; i < numSample; i++)
			result.add(rnd.nextInt(b));
		return result;
	}

	/**
	 * sampling with replacement based on a discrete cumulative distribution
	 * (cd)
	 * 
	 * @param numSample
	 * @param cd
	 * @return sampling of size numSample
	 */
	public static List<Integer> sample(int numSample, List<Double> cd) {
		List<Integer> result = new ArrayList<>();
		long seed = System.currentTimeMillis();
		Random rnd = new Random(seed);
		while (result.size() < numSample) {
			double rndVal = rnd.nextDouble();
			IntStream.range(0, cd.size() - 1).boxed().parallel().forEach(i -> {
				if (cd.get(i) < rndVal && rndVal < cd.get(i + 1))
					result.add(i);
			});
		}
		return result;
	}

	/**
	 * Sampling with replacement
	 * 
	 * @param numSample
	 * @param a
	 * @param b
	 * @return subset of size numSample from [l, u)
	 */
	public static List<Integer> sample(int numSample, int a, int b) {
		if (b <= a)
			throw new IllegalArgumentException("b must be larger than a");

		int len = b - a;
		long seed = System.currentTimeMillis();
		Random rand = new Random(seed);
		List<Integer> list = new ArrayList<>();
		for (int i = 0; i < numSample; i++)
			list.add(a + rand.nextInt(len + 1));
		return list;
	}

	/**
	 * Sampling without replacement
	 * 
	 * @param numSample
	 * @param b
	 * @return subset of size numSample from [0, b)
	 */
	public static List<Integer> sample(int numSample, int b) {
		return sample(numSample, 0, b);
	}

	public static List<double[]> rejectionSampling(int dim, int numSample) {
		List<double[]> samples = new ArrayList<>();
		double z = 1.0 / Math.pow(2 * Math.PI, dim / 2);
		UniformRealDistribution distribution = new UniformRealDistribution(0, 1),
				pDistribution = new UniformRealDistribution(0, z);

		while (samples.size() < numSample) {
			double[] sample = new double[dim];
			for (int i = 0; i < dim; i++)
				sample[i] = distribution.sample();
			if (samples.size() == 0)
				samples.add(sample);
			else {
				double p = calcP(samples, sample, z);
				if (pDistribution.sample() < p)
					samples.add(sample);
				else
					continue;
			}

			int idx = samples.size() - 1;
			if (reject(samples.get(idx)))
				samples.remove(idx);
		}
		return samples;
	}

	static boolean reject(double[] sample) {
		int dim = sample.length;
		double s = sample[0];
		for (int i = 1; i < dim; i++) {
			if (s < sample[i])
				return true;
			s = sample[i];
		}
		return false;
	}

	static double calcP(List<double[]> samples, double[] sample, double z) {
		double q = 0;
		int nSample = samples.size();
		for (double[] s : samples)
			q += Math.exp(-MathLib.Distance.euclidean(s, sample) / 2);
		q /= (nSample * z);
		return z - q;
	}

	public static double[] sampleOrdinalChain(int dim) {
		double[] sample = new double[dim];
		Random rnd = new Random(System.currentTimeMillis());
		sample[0] = rnd.nextDouble();
		UniformRealDistribution u = null;
		double sum = sample[0];
		for (int i = 1; i < dim; i++) {
			u = new UniformRealDistribution(0, sample[i - 1]);
			sum += (sample[i] = u.sample());
		}
		for (int i = 0; i < dim; i++)
			sample[i] /= sum;
		return sample;
	}
}
