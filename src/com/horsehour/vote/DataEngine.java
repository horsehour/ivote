package com.horsehour.vote;

import java.io.IOException;
import java.nio.file.CopyOption;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import org.apache.commons.collections4.iterators.PermutationIterator;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.commons.math3.distribution.EnumeratedDistribution;
import org.apache.commons.math3.util.CombinatoricsUtils;

import com.horsehour.util.Ace;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;
import com.horsehour.vote.axiom.MonotonicityCriterion;
import com.horsehour.vote.axiom.NeutralityCriterion;
import com.horsehour.vote.rule.Borda;
import com.horsehour.vote.rule.Bucklin;
import com.horsehour.vote.rule.Condorcet;
import com.horsehour.vote.rule.Coombs;
import com.horsehour.vote.rule.Copeland;
import com.horsehour.vote.rule.InstantRunoff;
import com.horsehour.vote.rule.KemenyYoung;
import com.horsehour.vote.rule.Maximin;
import com.horsehour.vote.rule.PairMargin;
import com.horsehour.vote.rule.Plurality;
import com.horsehour.vote.rule.RankedPairs;
import com.horsehour.vote.rule.Schulze;
import com.horsehour.vote.rule.Veto;
import com.horsehour.vote.rule.VotingRule;
import com.horsehour.vote.rule.multiseat.Position;
import com.horsehour.vote.rule.multiseat.PrefProfile;
import com.horsehour.vote.rule.multiseat.VoteLab;

/**
 * Preference engine generate preferences, preference profiles and extracting
 * features for voting rules learning
 * 
 * @author Chunheng Jiang
 * @version 3.0
 * @since Jun. 4, 2016
 */
public class DataEngine {
	/**
	 * Generate random permutation of n elements based on Knuth shuffles
	 * 
	 * @param m
	 * @return random permutation
	 */
	public static Integer[] getRandomPermutation(int m) {
		List<Integer> list = new ArrayList<>();
		for (int i = 0; i < m; i++)
			list.add(i);
		Collections.shuffle(list);

		Integer[] permutation = new Integer[m];
		list.toArray(permutation);
		for (int i = 0; i < m - 1; i++) {
			int j = MathLib.Rand.sample(0, m - i);
			int p = permutation[i];
			permutation[i] = permutation[i + j];
			permutation[i + j] = p;
		}
		return permutation;
	}

	/**
	 * 
	 * @param permutation
	 * @return random permutation based on an initial permutation
	 */
	public static Integer[] getRandomPermutation(Integer[] permutation) {
		int m = permutation.length;
		for (int i = 0; i < m - 1; i++) {
			int j = MathLib.Rand.sample(0, m - i);
			int p = permutation[i];
			permutation[i] = permutation[i + j];
			permutation[i + j] = p;
		}
		return permutation;
	}

	public static <T> List<T> getRandomPermutation(List<T> itemList) {
		int m = itemList.size();
		List<T> permutation = new ArrayList<>();
		for (int i = 0; i < m; i++)
			permutation.add(itemList.get(i));
		Collections.shuffle(permutation);

		for (int i = 0; i < m - 1; i++) {
			int j = MathLib.Rand.sample(0, m - i);
			T e = permutation.get(i);
			permutation.set(i, permutation.get(i + j));
			permutation.set(i + j, e);
		}
		return permutation;
	}

	/**
	 * Enumerate all possible permutations of items
	 * 
	 * @param itemList
	 * @return all possible permutations over the items in the list
	 */
	public static <T> List<List<T>> getAllPermutations(List<T> itemList) {
		PermutationIterator<T> iter = new PermutationIterator<>(itemList);
		List<List<T>> permutationList = new ArrayList<>();
		while (iter.hasNext())
			permutationList.add(iter.next());
		return permutationList;
	}

	/**
	 * Enumerate all possible permutations of m items
	 * 
	 * @param m
	 *            number of items
	 * @return all possible permutations of {0, 1, ... , m - 1}
	 */
	public static List<List<Integer>> getAllPermutations(int m) {
		List<Integer> itemList = IntStream.range(0, m).boxed().collect(Collectors.toList());
		PermutationIterator<Integer> iter = new PermutationIterator<>(itemList);
		List<List<Integer>> permutationList = new ArrayList<>();

		while (iter.hasNext())
			permutationList.add(iter.next());
		return permutationList;
	}

	/**
	 * @param numItem
	 *            number of items
	 * @param numVote
	 *            number of voters
	 * @param fpp
	 *            get full preference profiles over numItem by numVote (fpp =
	 *            true), anonymous equivalent classes of profiles (fpp = false)
	 * @return preference profiles in terms of stream
	 */
	static Stream<Profile<Integer>> getPreferenceProfiles(int numItem, int numVote, boolean fpp) {
		List<Integer> itemList = IntStream.range(0, numItem).boxed().collect(Collectors.toList());
		return getPreferenceProfiles(itemList, numVote, fpp);
	}

	public static Stream<Profile<Integer>> getAECPreferenceProfiles(int numItem, int numVote) {
		return getPreferenceProfiles(numItem, numVote, false);
	}

	public static Stream<Profile<Integer>> getFullPreferenceProfiles(int numItem, int numVote) {
		return getPreferenceProfiles(numItem, numVote, true);
	}

	/**
	 * @param numItem
	 *            number of items
	 * @param numVote
	 *            number of voters
	 * @param fpp
	 *            get full preference profiles over numItem by numVote (fpp =
	 *            true), anonymous equivalent classes of profiles (fpp = false)
	 * @return preference profiles in terms of stream
	 */
	static <T> Stream<Profile<T>> getPreferenceProfiles(List<T> itemList, int numVote, boolean fpp) {
		ProfileIterator<T> iter = new ProfileIterator<>(itemList, numVote, fpp);
		return StreamSupport.stream(Spliterators.spliteratorUnknownSize(iter, Spliterator.ORDERED), false);
	}

	public static <T> Stream<Profile<T>> getAECPreferenceProfiles(List<T> itemList, int numVote) {
		return getPreferenceProfiles(itemList, numVote, false);
	}

	public static <T> Stream<Profile<T>> getFullPreferenceProfiles(List<T> itemList, int numVote) {
		return getPreferenceProfiles(itemList, numVote, true);
	}

	/**
	 * Get all profiles labeled by the given voting rule, each entry has three
	 * components, including AEC profile, the winners based on the voting rule,
	 * and the size of the AEC profile
	 * 
	 * @param numItem
	 * @param numVote
	 * @param rule
	 * @return all profiles labeled by a voting rule
	 */
	public static Stream<ChoiceTriple<Integer>> getLabeledProfiles(int numItem, int numVote, VotingRule rule) {
		Stream<Profile<Integer>> profiles = getAECPreferenceProfiles(numItem, numVote);
		List<Long> stat = getAECStat(numItem, numVote);

		AtomicInteger count = new AtomicInteger(-1);
		Stream<ChoiceTriple<Integer>> result = null;
		result = profiles.map(profile -> {
			count.getAndIncrement();
			Long num = stat.get(count.intValue());
			List<Integer> winners = rule.getAllWinners(profile);
			if (winners == null)
				return null;
			return new ChoiceTriple<Integer>(profile, winners, num.intValue());
		}).filter(Objects::nonNull);
		return result;
	}

	/**
	 * Extract the features of a list of profiles, and labeled the input
	 * features with their rule-decided winners
	 * 
	 * @param profiles
	 * @param rule
	 *            expert rule in charge of selecting winners for the profiles
	 * @return a voting rule labeled data set
	 */
	public static <T> Stream<Pair<List<Float>, T>> getAllLabeledDataSet(Stream<Profile<T>> profiles, VotingRule rule) {
		Stream<Pair<List<Float>, T>> dataset = null;
		dataset = profiles.map(profile -> {
			List<Pair<List<Float>, T>> pairs = new ArrayList<>();
			for (T winner : rule.getAllWinners(profile))
				pairs.add(Pair.of(getFeatures(profile), winner));
			return pairs;
		}).filter(Objects::nonNull).flatMap(Collection::stream);
		return dataset;
	}

	/**
	 * @param numItem
	 * @param numVote
	 * @param rule
	 * @return List of labeled data, each of the triple entry contains three
	 *         components: feature vector, winners, and number of copies
	 */
	public static Stream<Triple<List<Float>, List<Integer>, Integer>> getLabeledAllDataSet(int numItem, int numVote,
			VotingRule rule) {
		Stream<Profile<Integer>> profiles = getAECPreferenceProfiles(numItem, numVote);
		List<Long> stat = DataEngine.getAECStat(numItem, numVote);

		AtomicInteger count = new AtomicInteger(0);
		Stream<Triple<List<Float>, List<Integer>, Integer>> dataset = null;
		dataset = profiles.map(profile -> {
			Long num = stat.get(count.intValue());
			List<Integer> winners = rule.getAllWinners(profile);
			if (winners == null)
				return null;
			return Triple.of(getFeatures(profile), winners, num.intValue());
		}).filter(Objects::nonNull);
		return dataset;
	}

	public static Stream<Pair<List<Float>, List<Integer>>> getLabeledAECDataSet(int numItem, int numVote,
			VotingRule rule) {
		Stream<Profile<Integer>> profiles = getAECPreferenceProfiles(numItem, numVote);

		Stream<Pair<List<Float>, List<Integer>>> dataset = null;
		dataset = profiles.map(profile -> {
			List<Integer> winners = rule.getAllWinners(profile);
			if (winners == null)
				return null;
			return Pair.of(getFeatures(profile), winners);
		}).filter(Objects::nonNull);
		return dataset;
	}

	/**
	 * @param profiles
	 * @return construct data set from list of triples
	 */
	public static Pair<double[][], int[]> getFlatDataSet(List<ChoiceTriple<Integer>> profiles) {
		int numSamples = profiles.stream().parallel().map(triple -> triple.getWinners().size() * triple.size())
				.reduce(0, Integer::sum);

		double[][] x = new double[numSamples][];
		int[] y = new int[numSamples];

		int count = 0;
		for (ChoiceTriple<Integer> triple : profiles) {
			Profile<Integer> profile = triple.getProfile();
			List<Integer> winners = triple.getWinners();

			List<Float> features = getFeatures(profile);
			double[] temp = new double[features.size()];
			for (int i = 0; i < features.size(); i++)
				temp[i] = features.get(i);

			int nCopy = triple.size();
			while (nCopy > 0) {
				for (int winner : winners) {
					x[count] = temp;
					y[count] = winner;
					count++;
				}
				nCopy--;
			}
		}
		return Pair.of(x, y);
	}

	/**
	 * @param profile
	 * @return Features extracted from profile
	 */
	public static <T> List<Float> getFeatures(Profile<T> profile) {
		T[] items = profile.getSortedItems();

		List<Float> features = new ArrayList<>();
		features.addAll(getFeatures(profile, items));

		// features.addAll(getPositionalFeatures(profile, items));
		// features.addAll(getPairwiseFeatures(profile, items));
		// features.addAll(getProfileSpaceFeatures(profile, items));
		// features.addAll(ensembleRuleFeatures(profile, items));
		// features.addAll(getProfileFeatures(profile, items));

		return features;
	}

	/**
	 * One profile contains many different profile preferences or permutations
	 * over items represent the preference relationship. Each element in the
	 * general profile feature vector is the count number of one possible
	 * preference ranking over the items.
	 * 
	 * @param profile
	 * @param items
	 * @return m!-dimensional general profile feature vector where m is the
	 *         number of different item in items
	 */
	public static <T> List<Float> getProfileSpaceFeatures(Profile<T> profile, T[] items) {
		List<List<T>> permutations = getAllPermutations(Arrays.asList(items));
		List<Float> features = new ArrayList<>();

		int numP = permutations.size();
		int[] hashCodes = new int[numP];
		for (int i = 0; i < numP; i++) {
			hashCodes[i] = permutations.get(i).hashCode();
			features.add(0F);
		}

		int numVote = profile.getNumVote();
		int count = 0;
		for (T[] pref : profile.data) {
			int index = ArrayUtils.indexOf(hashCodes, Arrays.deepHashCode(pref));
			features.set(index, profile.votes[count] * 1.0F / numVote);
			count++;
		}
		return features;
	}

	public static <T> List<Float> getProfileFeatures(Profile<T> profile, T[] items) {
		List<List<T>> permutations = getAllPermutations(Arrays.asList(items));

		int numP = permutations.size();
		float[] ratioP = new float[numP];
		int[] hashCodeP = new int[numP];
		for (int i = 0; i < numP; i++)
			hashCodeP[i] = permutations.get(i).hashCode();

		int count = 0, numVote = profile.getNumVote();
		for (T[] pref : profile.data) {
			int index = ArrayUtils.indexOf(hashCodeP, Arrays.deepHashCode(pref));
			ratioP[index] += profile.votes[count] * 1.0F / numVote;
			count++;
		}

		int numItem = items.length;
		int dim = numItem * numItem * numP;
		List<Float> features = new ArrayList<>(dim);
		for (int i = 0; i < dim; i++)
			features.add(0F);

		for (int i = 0; i < numP; i++) {// permutation id
			if (ratioP[i] == 0)// preference ranking nonexistent
				continue;

			List<T> permutation = permutations.get(i);
			for (int k = 0; k < numItem; k++) {// position
				int iid = ArrayUtils.indexOf(items, permutation.get(k));
				int fid = i + k * numP + iid * numItem * numP;
				features.set(fid, ratioP[i]);
			}
		}
		return features;
	}

	static <T> List<Float> getPositionalFeatures(Profile<T> profile, T[] items) {
		int numItem = items.length;
		int numVote = profile.getNumVote();

		int[][] positionalFeatures = new int[numItem][numItem];

		int k = 0;
		for (T[] pref : profile.data) {
			for (int i = 0; i < numItem; i++) {
				int itemI = ArrayUtils.indexOf(items, pref[i]);
				positionalFeatures[itemI][i] += profile.votes[k];
			}
			k++;
		}
		List<Float> features = new ArrayList<>();
		for (int[] votes : positionalFeatures)
			for (int vote : votes)
				features.add(vote * 1.0F / numVote);
		return features;
	}

	static <T> List<Float> ensembleRuleFeatures(Profile<T> profile, T[] items) {
		List<VotingRule> rules = new ArrayList<>();
		rules.add(new Borda());
		rules.add(new Condorcet());
		rules.add(new Coombs());
		rules.add(new Copeland());
		rules.add(new InstantRunoff());
		rules.add(new KemenyYoung());
		rules.add(new Maximin());
		rules.add(new PairMargin());
		rules.add(new Plurality());
		rules.add(new RankedPairs());
		rules.add(new Schulze());
		rules.add(new Veto());
		rules.add(new Bucklin());

		List<Float> features = new ArrayList<>();
		List<T> winners = null;

		for (VotingRule rule : rules) {
			winners = rule.getAllWinners(profile);
			addMatchedItems(items, winners, features);
		}
		return features;
	}

	static <T> void addMatchedItems(T[] items, List<T> winners, List<Float> features) {
		if (winners == null)
			for (int i = 0; i < items.length; i++)
				features.add(0.0F);
		else
			for (T item : items) {
				int i = winners.indexOf(item);
				if (i == -1)
					features.add(0.0F);
				else
					features.add(1.0F);
			}
	}

	/**
	 * Pairwise Features of Preference Profiles
	 * 
	 * @param profile
	 * @param items
	 * @return Pairwise features without the invalid zero-elements
	 */
	static <T> List<Float> getPairwiseFeatures(Profile<T> profile, T[] items) {
		int[][] ppm = Condorcet.getPairwisePreferenceMatrix(profile, items);
		int numVote = profile.getNumVote();
		List<Float> features = new ArrayList<>();
		// pairwise comparison features
		for (int[] wins : ppm)
			for (int win : wins)
				features.add(win * 1.0F / numVote);
		return features;
	}

	/**
	 * Extract all general profile features
	 * 
	 * @param profile
	 * @param items
	 * @return the number of voters who places one item in position i, and
	 *         another one in position j
	 */
	static <T> List<float[][]> getGeneralFeatures(Profile<T> profile, T[] items) {
		if (items == null)
			items = profile.getSortedItems();

		int numItem = items.length;
		int numPreferences = profile.data.length;
		int[][] indices = new int[numItem][numPreferences];

		for (int i = 0; i < numItem; i++)
			for (int k = 0; k < numPreferences; k++) {
				T[] preferences = profile.data[k];
				indices[i][k] = ArrayUtils.indexOf(preferences, items[i]);
			}

		int nv = profile.numVoteTotal;
		List<float[][]> features = new ArrayList<>();
		for (int i = 0; i < numItem; i++) {
			for (int j = i + 1; j < numItem; j++) {
				float[][] pairFeature = new float[numItem][numItem];
				for (int k = 0; k < numPreferences; k++) {
					int posI = indices[i][k];
					int posJ = indices[j][k];
					pairFeature[posI][posJ] += (profile.votes[k] * 1.0F / nv);
				}
				features.add(pairFeature);
			}
		}
		return features;
	}

	static <T> List<Float> getFeatures(Profile<T> profile, T[] items) {
		if (items == null)
			items = profile.getSortedItems();

		int numItem = items.length;
		int numPreferences = profile.data.length;
		int[][] positions = new int[numItem][numPreferences];

		for (int i = 0; i < numItem; i++) {
			for (int k = 0; k < numPreferences; k++) {
				T[] preferences = profile.data[k];
				positions[i][k] = ArrayUtils.indexOf(preferences, items[i]);
			}
		}

		int nv = profile.numVoteTotal;
		List<Float> features = new ArrayList<>();
		for (int i = 0; i < numItem; i++) {
			for (int j = i + 1; j < numItem; j++) {
				float[][] pairFeature = new float[numItem][numItem];
				for (int k = 0; k < numPreferences; k++) {
					int posI = positions[i][k];
					int posJ = positions[j][k];

					pairFeature[posI][posJ] += (profile.votes[k] * 1.0F / nv);
				}
				for (int idxI = 0; idxI < numItem; idxI++)
					for (int idxJ = 0; idxJ < numItem; idxJ++) {
						if (idxI == idxJ)
							continue;

						features.add(pairFeature[idxI][idxJ]);
					}
			}
		}
		return features;
	}

	public static double[][] getFeatures(Profile<Integer> profile, List<Integer> items) {
		int numItem = items.size();
		int nv = profile.numVoteTotal;
		int numPreferences = profile.data.length;
		int[][] positions = new int[numItem][numPreferences];

		double[][] features = new double[numItem][4];
		// sum of Borda scores
		double sum = 0;

		for (int k = 0; k < numPreferences; k++) {
			Integer[] preferences = profile.data[k];
			int c = 0;
			for (int p = 0; p < preferences.length; p++) {
				int idx = -1;
				if ((idx = items.indexOf(preferences[p])) > -1) {
					c++;
					// Borda score
					features[idx][0] += profile.votes[k] * (numItem - c);
					// normalized plurality score
					features[idx][1] += (c == 1 ? (profile.votes[k] * 1.0F / nv) : 0);
					// normalized veto score
					features[idx][3] += (c == numItem ? (profile.votes[k] * 1.0F / nv) : 0);
					sum += features[idx][0];
					positions[idx][k] = c;
				}
			}
		}

		// normalized winning rate
		for (int i = 0; i < numItem; i++) {
			// normalized Borda score
			features[i][0] /= sum;
			for (int j = i + 1; j < numItem; j++) {
				for (int k = 0; k < numPreferences; k++) {
					float weight = (profile.votes[k] * 1.0F / nv) / (numItem - 1);
					if (positions[i][k] < positions[j][k])
						features[i][2] += weight;
					else
						features[j][2] += weight;
				}
			}
		}
		return features;
	}

	/**
	 * Each anonymous equivalent class (AEC) includes many equivalent named
	 * profiles. For an AEC {2 * [1], 1 * [2]}, where the numbers in the square
	 * are the indices of the basis preference rankings. The AEC represents 3
	 * preference profiles {112}, {121} and {211}, and 3 = C(3, 2) * C(1, 1).
	 * Suppose we have m items, there are m! permutations over them. Again,
	 * suppose that there are n voters, each of whom cast a strict linear
	 * preference rankings of m items/candidates, with the order of each
	 * preference in consideration, there are (m!)^n preference profiles in
	 * total. When the names of voters are anonymous, we could divide the
	 * profiles into C(n + m! - 1, m! - 1) AECs. For example, the AEC {n_1 *
	 * [R_1], n_2 * [R_2], ..., n_(m!)* [R_(m!)]}, where n = n_1 + n_2 + ... +
	 * n(m!). There are C(m!, n_1) * C(m! - n_1, n_2) * C(n - n_1 - n_2, n_3) *
	 * ... * C(n_(m!), n_(m!)) = n!/[(n_1!)(n_2!)...(n_(m!)!)] profiles
	 * belonging to this same AEC. Based on the formula, it's possible to
	 * compute each AEC's probability with the percentage of its member size to
	 * the total number of named preference profiles. The result could
	 * facilitate to sample preference profiles.
	 * 
	 * @param aecProfile
	 *            anonymous equivalent class preference profile
	 * @param mFactorial
	 *            m factorial - m!
	 * @return probability of each AEC in the full preference profiles
	 */
	public static <E> double getAECProb(Profile<E> aecProfile, Long mFactorial) {
		int sum = 0;
		double prod = 1.0d;
		for (int vote : aecProfile.votes) {
			int remain = aecProfile.numVoteTotal - sum;
			sum += vote;
			prod *= (CombinatoricsUtils.binomialCoefficient(remain, vote) / Math.pow(mFactorial, vote));
		}
		return prod;
	}

	/**
	 * @param itemList
	 * @param numVote
	 * @return Probability distribution of AEC
	 */
	public static <E> List<Double> getAECDistribution(List<E> itemList, int numVote) {
		ProfileIterator<E> iter = new ProfileIterator<>(itemList, numVote, false);
		Long mFactorial = CombinatoricsUtils.factorial(itemList.size());
		List<Double> distribution = new ArrayList<>();
		while (iter.hasNext())
			distribution.add(getAECProb(iter.next(), mFactorial));
		return distribution;
	}

	public static EnumeratedDistribution<Integer> getAECDistribution(int numItem, int numVote) {
		Long mFactorial = CombinatoricsUtils.factorial(numItem);
		List<Integer> permutations = IntStream.range(0, mFactorial.intValue()).boxed().collect(Collectors.toList());
		Selection<Integer> iter = new Selection<>(permutations, numVote, false);
		List<org.apache.commons.math3.util.Pair<Integer, Double>> distribution = new ArrayList<>();
		int count = 0;
		while (iter.hasNext()) {
			distribution.add(org.apache.commons.math3.util.Pair.create(count, getAECProb(iter.next(), mFactorial)));
			count++;
		}
		return new EnumeratedDistribution<Integer>(distribution);
	}

	public static List<Long> getAECStat(int numItem, int numVote) {
		List<Integer> itemList = IntStream.range(0, numItem).boxed().collect(Collectors.toList());
		ProfileIterator<Integer> iter = new ProfileIterator<>(itemList, numVote, false);

		List<Long> stat = new ArrayList<>();

		Profile<Integer> aecProfile;
		while (iter.hasNext()) {
			aecProfile = iter.next();
			int sum = 0;
			long prod = 1;
			for (int vote : aecProfile.votes) {
				int remain = aecProfile.numVoteTotal - sum;
				sum += vote;
				prod *= CombinatoricsUtils.binomialCoefficient(remain, vote);
			}
			stat.add(prod);
		}
		return stat;
	}

	public static long getNumAEC(int numItem, int numVote) {
		List<Integer> itemList = IntStream.range(0, numItem).boxed().collect(Collectors.toList());
		ProfileIterator<Integer> iter = new ProfileIterator<>(itemList, numVote, false);
		long num = 0;
		while (iter.hasNext()) {
			iter.next();
			num++;
		}
		return num;
	}

	/**
	 * Probability of Condorcet winner consistent AEC
	 * 
	 * @param numItem
	 * @param numVote
	 * @return Probability of CWC-AEC
	 */
	public static EnumeratedDistribution<Integer> getCWCAECDistribution(int numItem, int numVote) {
		List<Integer> itemList = IntStream.range(0, numItem).boxed().collect(Collectors.toList());
		ProfileIterator<Integer> iter = new ProfileIterator<>(itemList, numVote, false);
		Long mFactorial = CombinatoricsUtils.factorial(itemList.size());
		List<org.apache.commons.math3.util.Pair<Integer, Double>> distribution = new ArrayList<>();

		Condorcet condorcet = new Condorcet();
		Profile<Integer> profile;
		int count = -1;
		while (iter.hasNext()) {
			count++;
			profile = iter.next();
			List<Integer> cw = condorcet.getAllWinners(profile);
			if (cw == null)
				continue;
			distribution.add(org.apache.commons.math3.util.Pair.create(count, getAECProb(profile, mFactorial)));
		}
		return new EnumeratedDistribution<>(distribution);
	}

	/**
	 * Perform the computing directly on AEC index list
	 * 
	 * @param aecIndexList
	 * @param mFactorial
	 * @return Probability of AEC
	 */
	static double getAECProb(List<Integer> aecIndexList, Long mFactorial) {
		Map<Integer, Long> map = aecIndexList.stream().collect(Collectors.groupingBy(e -> e, Collectors.counting()));
		int sum = 0, count = 0;
		double prod = 1.0d;
		for (int key : map.keySet()) {
			int remain = aecIndexList.size() - sum;
			count = map.get(key).intValue();
			sum += count;
			prod *= (CombinatoricsUtils.binomialCoefficient(remain, count) / Math.pow(mFactorial, count));
		}
		return prod;
	}

	/**
	 * Sampling preference profiles according to the distribution of the
	 * permutations of the candidates. More specifically, there are m!
	 * permutations, each preference profile constitutes of n preferences from
	 * the m! permutations. Suppose permutations are equiproable, that p_1 = p_2
	 * = ... = p_{m!} = 1/m!, and the generating of a preference profile which
	 * contains n preferences actually could be considered as a multinomial
	 * trials process.
	 * 
	 * @param numItem
	 * @param numVote
	 * @param numSample
	 * @return random preference profiles
	 */
	public static List<Profile<Integer>> getRandomProfiles(int numItem, int numVote, int numSample) {
		List<List<Integer>> permutations = DataEngine.getAllPermutations(numItem);
		int nPermutation = permutations.size();

		/**
		 * equiproable
		 */
		double[] prob = MathLib.Matrix.ones(nPermutation);

		List<Profile<Integer>> profiles = new ArrayList<>(numSample);
		for (int[] sample : MathLib.Rand.getMultinomialSamples(prob, numVote, numSample)) {
			int count = 0;
			for (int i = 0; i < nPermutation; i++) {
				if (sample[i] == 0)
					continue;
				count++;// non-zero values
			}

			if (count == nPermutation)
				profiles.add(new Profile<>(permutations, sample));
			else {
				int[] votes = new int[count];
				int index = -1;
				Integer[][] preferences = new Integer[count][numItem];
				for (int i = 0; i < nPermutation; i++) {
					if (sample[i] == 0)
						continue;

					index++;
					votes[index] = sample[i];
					permutations.get(i).toArray(preferences[index]);
				}
				profiles.add(new Profile<>(preferences, votes));
			}
		}
		return profiles;
	}

	/**
	 * @param numItem
	 * @param numVote
	 * @return generate random profile based on Knuth shuffles
	 */
	public static Profile<Integer> getRandomProfile(int numItem, int numVote) {
		Integer[][] data = new Integer[numVote][];
		for (int i = 0; i < numVote; i++)
			data[i] = getRandomPermutation(numItem);
		Profile<Integer> profile = new Profile<>(data);
		profile = profile.compress();
		return profile;
	}

	/**
	 * visualize the simulation approximation of the random sampling method to
	 * the exact distribution of preference profiles
	 */
	public static void simulatedApproximation(int numItem, int numVote, int numSample) {
		Long nPermutation = MathLib.Data.factorial(numItem);
		double[] prob = MathLib.Matrix.ones(nPermutation.intValue());

		/**
		 * row - sample index, column - votes of corresponding preference
		 * rankings / permutations, zero means that no vote cast to the
		 * preference ranking.
		 */
		int[][] samples = MathLib.Rand.getMultinomialSamples(prob, numVote, numSample);

		Map<String, Long> countTable = Arrays.stream(samples).map(Arrays::toString)
				.collect(Collectors.groupingBy(e -> e, Collectors.counting()));

		double[][] stat = new double[2][countTable.size()];
		int i = 0;
		for (long count : countTable.values())
			stat[0][i++] = count;

		i = 0;
		for (String key : countTable.keySet()) {
			key = key.replace("[", "").replace("]", "");// parse votes
			stat[1][i++] = new Profile<>(new Integer[0][],
					Arrays.stream(key.split(", ")).mapToInt(Integer::parseInt).toArray()).getStat()
					/ Math.pow(nPermutation, numVote) * numSample;
		}

		List<String> columnLabel = new ArrayList<>(countTable.size());
		i = 0;
		for (; i < countTable.size(); i++)
			columnLabel.add("P" + (i + 1));// preference profile index

		Ace ace = new Ace("Preference Profile Sampling");
		ace.bar(Arrays.asList("Sampling Profile", "Truth Profile"), columnLabel, stat);
	}

	public static Stream<ChoiceTriple<Integer>> getAllLabeledProfiles(int numItem, int numVote, VotingRule rule) {
		Stream<Profile<Integer>> profiles = getAECPreferenceProfiles(numItem, numVote);

		AtomicInteger count = new AtomicInteger(-1);
		Stream<ChoiceTriple<Integer>> result = null;
		result = profiles.map(profile -> {
			count.getAndIncrement();
			List<Integer> winners = rule.getAllWinners(profile);
			if (winners == null || winners.size() > 1)
				return null;
			return new ChoiceTriple<>(profile, winners, 1);
		}).filter(Objects::nonNull);
		return result;
	}

	/**
	 * @param numItem
	 * @param numVote
	 * @param numSample
	 * @param rule
	 * @return Randomly labeled profiles using a specific voting rule
	 */
	public static List<ChoiceTriple<Integer>> getRandomLabeledProfiles(int numItem, int numVote, int numSample,
			VotingRule rule) {
		Stream<Profile<Integer>> profiles = getAECPreferenceProfiles(numItem, numVote);
		EnumeratedDistribution<Integer> distribution = null;
		if (rule instanceof Condorcet)
			distribution = getCWCAECDistribution(numItem, numVote);
		else
			distribution = getAECDistribution(numItem, numVote);

		Integer[] sampleAECProfile = new Integer[numSample];
		// sampling according to distribution
		distribution.sample(numSample, sampleAECProfile);

		// redundancies number of samples
		Map<Integer, Long> countTable = Arrays.asList(sampleAECProfile).stream()
				.collect(Collectors.groupingBy(e -> e, Collectors.counting()));

		Random rnd = new Random(System.currentTimeMillis());
		AtomicInteger index = new AtomicInteger(-1), count = new AtomicInteger(0);
		Stream<ChoiceTriple<Integer>> result = null;
		result = profiles.map(profile -> {
			index.getAndIncrement();
			if (!countTable.containsKey(index.intValue()))
				return null;

			Long num = countTable.get(index.intValue());
			List<Integer> winners = rule.getAllWinners(profile);
			if (winners == null || winners.size() > 1)
				return null;

			if (winners.size() == 1)
				count.addAndGet(num.intValue());
			return new ChoiceTriple<>(profile, winners, num.intValue());
		}).filter(Objects::nonNull);

		List<ChoiceTriple<Integer>> samples = result.collect(Collectors.toList());

		while (count.get() < numSample) {
			int idx = rnd.nextInt(samples.size());
			ChoiceTriple<Integer> triple = samples.get(idx);
			samples.set(idx, new ChoiceTriple<>(triple.getProfile(), triple.getWinners(), triple.size() + 1));
			count.getAndIncrement();
		}
		return samples;
	}

	/**
	 * @param numItem
	 * @param rule
	 * @return Base profiles consisting of all permutations of m items
	 */
	public static List<ChoiceTriple<Integer>> getLabeledBaseProfiles(int numItem, VotingRule rule) {
		List<List<Integer>> permutations = getAllPermutations(numItem);
		List<ChoiceTriple<Integer>> samples = new ArrayList<>();
		for (List<Integer> perm : permutations) {
			Integer[][] data = new Integer[1][numItem];
			perm.toArray(data[0]);
			Profile<Integer> profile = new Profile<>(data);
			List<Integer> winners = rule.getAllWinners(profile);
			if (winners == null || winners.size() > 1)
				continue;
			else
				samples.add(new ChoiceTriple<>(profile, winners, 1));
		}
		return samples;
	}

	/**
	 * Condorcet profiles could be catched using Condorcet voting rule
	 * 
	 * @param numItem
	 * @param numVote
	 * @param numSample
	 * @return random condorcet profiles
	 */
	public static List<ChoiceTriple<Integer>> getRandomCondorcetProfiles(int numItem, int numVote, int numSample) {
		return getRandomLabeledProfiles(numItem, numVote, numSample, new Condorcet());
	}

	/**
	 * @param numItem
	 * @param numVote
	 * @return random profile which has an unique Condorcet winner
	 */
	public static ChoiceTriple<Integer> getRandomCondorcetProfile(int numItem, int numVote) {
		Profile<Integer> profile = DataEngine.getRandomProfile(numItem, numVote);
		VotingRule rule = new Condorcet();

		List<Integer> winners = null;
		while ((winners = rule.getAllWinners(profile)) == null)
			profile = DataEngine.getRandomProfile(numItem, numVote);
		return new ChoiceTriple<>(profile, winners, 1);
	}

	/**
	 * Generate random neutral profiles which based on the neutrality criterion.
	 * There are two sources for the profiles generating. It randomly selects
	 * many root/base profiles, and their winners respectively and performs all
	 * possible permutations on each of those profiles, and their winners
	 * therein. The procedure produces at least k * (m! - 1) samples if the
	 * number of root profiles is k and each of those profiles have only one
	 * winner.
	 * 
	 * @param baseProfiles
	 * @param numSample
	 * @return neutral profiles from specific source profiles
	 */
	public static List<ChoiceTriple<Integer>> getRandomNeutralProfiles(List<List<ChoiceTriple<Integer>>> baseProfiles,
			int numSample) {
		List<ChoiceTriple<Integer>> baseProfileList = baseProfiles.stream().flatMap(profiles -> profiles.stream())
				.collect(Collectors.toList());

		List<Integer> copyList = new ArrayList<>();
		for (ChoiceTriple<Integer> triple : baseProfileList)
			copyList.add(triple.size());

		List<org.apache.commons.math3.util.Pair<Integer, Double>> pairList = new ArrayList<>();
		for (int i = 0; i < copyList.size(); i++)
			pairList.add(org.apache.commons.math3.util.Pair.create(i, 1.0 * copyList.get(i)));

		EnumeratedDistribution<Integer> distribution = new EnumeratedDistribution<>(pairList);

		int numItem = baseProfiles.get(0).get(0).getProfile().getNumItem();
		// Long numFactorial = MathUtils.factorial(numItem);
		// numSample /= numFactorial.intValue();

		Integer[] samples = new Integer[numSample];
		// sampling according to distribution
		distribution.sample(numSample, samples);

		Map<Integer, Long> countTable = Arrays.asList(samples).stream()
				.collect(Collectors.groupingBy(e -> e, Collectors.counting()));

		NeutralityCriterion criterion = new NeutralityCriterion();
		List<List<Integer>> permutations = getAllPermutations(numItem);

		int count = -1, numCollect = 0;
		List<ChoiceTriple<Integer>> result = new ArrayList<>();

		for (ChoiceTriple<Integer> triple : baseProfileList) {
			count++;
			if (!countTable.containsKey(count))
				continue;

			Profile<Integer> profile = triple.getProfile();
			Long num = countTable.get(count);
			List<Integer> winners = triple.getWinners();
			for (Pair<Profile<Integer>, List<Integer>> pair : criterion.getAllPermutedProfiles(profile, winners,
					permutations)) {
				result.add(new ChoiceTriple<>(pair.getLeft(), pair.getRight(), num.intValue()));
				numCollect += num;
				if (numCollect >= numSample)
					return result;
			}
		}
		return result;
	}

	/**
	 * Based on the base profiles from which we generate specific number of
	 * monotonic profiles.
	 * 
	 * @param baseProfiles
	 * @param numSample
	 * @return monotonic profiles and related labels
	 */
	public static List<ChoiceTriple<Integer>> getRandomMonotonicProfiles(List<List<ChoiceTriple<Integer>>> baseProfiles,
			int numSample) {

		List<ChoiceTriple<Integer>> baseProfileList = baseProfiles.stream().flatMap(profiles -> profiles.stream())
				.collect(Collectors.toList());

		List<Integer> copyList = new ArrayList<>();
		for (ChoiceTriple<Integer> triple : baseProfileList)
			copyList.add(triple.size());

		List<org.apache.commons.math3.util.Pair<Integer, Double>> pairList = new ArrayList<>();
		for (int i = 0; i < copyList.size(); i++)
			pairList.add(org.apache.commons.math3.util.Pair.create(i, 1.0 * copyList.get(i)));

		EnumeratedDistribution<Integer> distribution = new EnumeratedDistribution<>(pairList);

		// int numItem = baseProfiles.get(0).get(0).getLeft().getNumItem();
		// int numVote = baseProfiles.get(0).get(0).getLeft().numVote;
		// numSample = 2 * numSample / ((numItem - 1) * numVote);

		Integer[] samples = new Integer[numSample];
		distribution.sample(numSample, samples);

		Map<Integer, Long> countTable = Arrays.asList(samples).stream()
				.collect(Collectors.groupingBy(e -> e, Collectors.counting()));

		MonotonicityCriterion criterion = new MonotonicityCriterion();

		int count = -1, numCollect = 0;
		List<ChoiceTriple<Integer>> result = new ArrayList<>();

		for (ChoiceTriple<Integer> triple : baseProfileList) {
			count++;
			if (!countTable.containsKey(count))
				continue;

			Long num = countTable.get(count);
			Profile<Integer> profile = triple.getProfile();
			List<Integer> winners = triple.getWinners();
			for (ChoiceTriple<Integer> t : criterion.getAllRaisedProfiles(profile, winners)) {
				result.add(new ChoiceTriple<>(t.getProfile(), t.getWinners(), t.size() * num.intValue()));
				numCollect += t.size() * num;
				if (numCollect >= numSample)
					return result;
			}
		}
		return result;
	}

	/**
	 * Based on the base profiles from which we generate specific number of
	 * consistent profiles.
	 * 
	 * @param baseProfiles
	 * @param numSample
	 * @return consistent profiles and related labels
	 */
	public static List<ChoiceTriple<Integer>> getRandomConsistentProfiles(
			List<List<ChoiceTriple<Integer>>> baseProfiles, int numSample) {

		List<ChoiceTriple<Integer>> profiles = baseProfiles.stream().flatMap(p -> p.stream())
				.collect(Collectors.toList());

		// (winner, index list)
		Map<List<Integer>, List<Integer>> cluster = IntStream.range(0, profiles.size()).boxed()
				.collect(Collectors.groupingBy(i -> profiles.get(i).getWinners(), Collectors.toList()));

		List<ChoiceTriple<Integer>> result = new ArrayList<>();
		/**
		 * To make sure all winners have the chance being included, we build
		 * random sampling pool from all possible combinations of profiles
		 */
		List<org.apache.commons.math3.util.Pair<Integer, Double>> pairList = new ArrayList<>();
		int count = 0;
		for (List<Integer> winners : cluster.keySet()) {
			List<Integer> pidList = cluster.get(winners);
			for (int i = 0; i < pidList.size(); i++) {
				int pidI = pidList.get(i);
				int nCopyI = profiles.get(pidI).size();
				for (int j = i; j < pidList.size(); j++) {
					int pidJ = pidList.get(j);
					int nCopyJ = profiles.get(pidJ).size();
					pairList.add(org.apache.commons.math3.util.Pair.create(count, 1.0 * nCopyI * nCopyJ));
					count++;
				}
			}
		}
		EnumeratedDistribution<Integer> distribution = new EnumeratedDistribution<>(pairList);
		Integer[] samples = new Integer[numSample];
		distribution.sample(numSample, samples);

		Map<Integer, Long> countTable = Arrays.asList(samples).stream()
				.collect(Collectors.groupingBy(e -> e, Collectors.counting()));

		count = 0;
		for (List<Integer> winners : cluster.keySet()) {
			List<Integer> pidList = cluster.get(winners);
			for (int i = 0; i < pidList.size(); i++) {
				int pidI = pidList.get(i);
				for (int j = i; j < pidList.size(); j++) {
					int pidJ = pidList.get(j);
					if (countTable.containsKey(count)) {
						result.add(new ChoiceTriple<>(
								profiles.get(pidI).getProfile().merge(profiles.get(pidJ).getProfile()), winners,
								countTable.get(count).intValue()));
					}
					count++;
				}
			}
		}
		return result;
	}

	@SafeVarargs
	public static Stream<ChoiceTriple<Integer>> mergeProfiles(Stream<ChoiceTriple<Integer>>... sources) {
		Stream<ChoiceTriple<Integer>> stream = Stream.empty();
		for (int i = 0; i < sources.length; i++)
			stream = Stream.concat(stream, sources[i]);
		return stream;
	}

	public static void getSVMDataSet(List<ChoiceTriple<Integer>> profiles, String dest) throws IOException {
		Pair<double[][], int[]> trainset = DataEngine.getFlatDataSet(profiles);
		double[][] input = trainset.getLeft();
		int[] y = trainset.getRight();

		int n = y.length, d = input[0].length;
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < n; i++) {
			sb.append(y[i] + 1);
			double[] x = input[i];
			for (int k = 0; k < d; k++)
				sb.append(" " + (k + 1) + ":" + x[k]);
			sb.append("\n");
		}
		OpenOption[] options = { StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE };
		Files.write(Paths.get(dest), sb.toString().getBytes(), options);
	}

	/**
	 * Generate SVM training set
	 * 
	 * @param numItem
	 * @param numVote
	 * @param numSample
	 * @param oracle
	 * @param numVotes
	 * @param dest
	 * @throws IOException
	 */
	public static void getSVMTrainingSet(int numItem, int numVote, int numSample, VotingRule oracle, String dest)
			throws IOException {
		List<ChoiceTriple<Integer>> profiles = DataEngine.getRandomLabeledProfiles(numItem, numVote, numSample, oracle);
		getSVMDataSet(profiles, dest);
	}

	/**
	 * Generate SVM testing set
	 * 
	 * @param numItem
	 * @param numVote
	 * @param oracle
	 * @param testFile
	 * @param truthList
	 * @throws IOException
	 */
	public static void getSVMTestingSet(int numItem, int numVote, VotingRule oracle, String testFile,
			List<Integer> truthList) throws IOException {
		Stream<Profile<Integer>> profiles = DataEngine.getAECPreferenceProfiles(numItem, numVote);

		StringBuffer sb = new StringBuffer();
		for (Profile<Integer> profile : profiles.collect(Collectors.toList())) {
			List<Integer> winners = oracle.getAllWinners(profile);
			// single winner
			if (winners == null || winners.size() > 1)
				continue;

			List<Float> feature = DataEngine.getFeatures(profile);
			int label = winners.get(0) + 1;
			sb.append(label);
			for (int i = 0; i < feature.size(); i++)
				sb.append(" " + (i + 1) + ":" + feature.get(i));
			sb.append("\n");
			truthList.add(label);
		}
		OpenOption[] options = { StandardOpenOption.CREATE, StandardOpenOption.WRITE };
		Files.write(Paths.get(testFile), sb.toString().getBytes(), options);
	}

	/**
	 * Parse profile from soc (strictly ordering complete list) file provided on
	 * PrefLib
	 * 
	 * @param socFile
	 * @return profiles
	 */
	public static Profile<Integer> getSOCProfile(String socFile) {
		List<String> lines = null;
		try {
			lines = Files.readAllLines(Paths.get(socFile));
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}

		int numItem = Integer.parseInt(lines.get(0));
		int n = lines.size() - numItem - 2;
		Integer[][] data = new Integer[n][numItem];
		int[] votes = new int[n];
		String[] fields = null;
		int count = 0;
		for (int i = numItem + 2; i < lines.size(); i++) {
			fields = lines.get(i).split(",");
			if (fields.length == numItem + 1) {
				votes[count] = Integer.parseInt(fields[0]);
				for (int k = 1; k < fields.length; k++)
					data[count][k - 1] = Integer.parseInt(fields[k]);
			}
			count++;
		}
		return new Profile<>(data, votes);
	}

	/**
	 * Generate the full profile space with each possible rankings over specific
	 * number of candidates getting one vote
	 * 
	 * @param numItem
	 * @param baseFile
	 * @throws IOException
	 */
	public static void generateFullProfileSpace(int numItem, String baseFile) throws IOException {
		StringBuffer sb = new StringBuffer();
		long np = MathLib.Data.factorial(numItem);
		sb.append(numItem).append("\n");
		for (int i = 0; i < numItem; i++)
			sb.append(i + ",c" + i + "\n");
		sb.append(np + "," + np + "," + np + "\n");

		List<Integer> itemList = IntStream.range(0, numItem).boxed().collect(Collectors.toList());
		PermutationIterator<Integer> iter = new PermutationIterator<>(itemList);

		String dest = baseFile + "M" + numItem + ".csv";
		OpenOption[] options = { StandardOpenOption.CREATE, StandardOpenOption.APPEND, StandardOpenOption.WRITE };

		List<Integer> permutation = null;
		int count = 0, nBatch = 5000;
		while (iter.hasNext()) {
			permutation = iter.next();
			sb.append("1,");
			sb.append(permutation.stream().map(i -> i.toString()).collect(Collectors.joining(",")));
			sb.append("\n");

			if (count > 0 && count == nBatch) {
				Files.write(Paths.get(dest), sb.toString().getBytes(), options);
				sb = new StringBuffer();
				count = 0;
			} else
				count++;
		}

		if (count > 0) {
			Files.write(Paths.get(dest), sb.toString().getBytes(), options);
		}
	}

	/**
	 * Generate the complete profile space given m and n
	 * 
	 * @param numItem
	 * @param numVote
	 * @param baseFile
	 */
	public static void generateFullProfileSpace(int numItem, int numVote, String baseFile) {
		StringBuffer meta = new StringBuffer();
		meta.append(numItem + "\n");
		for (int i = 0; i < numItem; i++)
			meta.append(i + ",c" + i + "\n");
		OpenOption[] options = { StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING,
				StandardOpenOption.WRITE };

		List<Integer> items = new ArrayList<>();
		for (int i = 0; i < numItem; i++)
			items.add(i);

		Stream<Profile<Integer>> stream = DataEngine.getPreferenceProfiles(numItem, numVote, false);

		AtomicLong count = new AtomicLong(0);
		stream.forEach(profile -> {
			count.incrementAndGet();
			profile = profile.compress();
			StringBuffer sb = new StringBuffer();
			sb.append(meta);
			sb.append(numVote + "," + numVote + "," + profile.data.length + "\n");

			Integer[][] data = profile.data;
			for (int i = 0; i < data.length; i++) {
				sb.append(profile.votes[i]);
				for (int k = 0; k < data[i].length; k++)
					sb.append("," + data[i][k]);
				sb.append("\n");
			}
			String name = "M" + numItem + "N" + numVote + "-" + count + ".csv";
			try {
				String socFile = baseFile + "/" + name;
				Files.write(Paths.get(socFile), sb.toString().getBytes(), options);
			} catch (IOException e) {
				e.printStackTrace();
				return;
			}
		});
	}

	/**
	 * Generate random profile spaces for given m, n and num of samples
	 * 
	 * @param numItem
	 * @param numVote
	 * @param numSample
	 * @param baseFile
	 */
	public static void generateRandomProfiles(int numItem, int numVote, int numSample, String baseFile) {
		StringBuffer meta = new StringBuffer();
		meta.append(numItem + "\n");
		for (int i = 0; i < numItem; i++)
			meta.append(i + ",c" + i + "\n");

		StringBuffer sb = null;
		Integer[][] data = null;
		String socFile = "";
		OpenOption[] options = { StandardOpenOption.CREATE, StandardOpenOption.WRITE,
				StandardOpenOption.TRUNCATE_EXISTING };

		Profile<Integer> profile = null;

		for (int s = 1; s <= numSample; s++) {
			profile = DataEngine.getRandomProfile(numItem, numVote);
			sb = new StringBuffer();
			sb.append(meta);
			sb.append(numVote + "," + numVote + "," + profile.data.length + "\n");

			data = profile.data;
			for (int i = 0; i < data.length; i++) {
				sb.append(profile.votes[i]);
				for (int k = 0; k < data[i].length; k++)
					sb.append("," + data[i][k]);
				sb.append("\n");
			}

			String name = "M" + numItem + "N" + numVote + "-" + s + ".csv";
			try {
				socFile = baseFile + "/" + name;
				Files.write(Paths.get(socFile), sb.toString().getBytes(), options);
			} catch (IOException e) {
				e.printStackTrace();
				return;
			}
		}
	}

	/**
	 * load preference profile from local file
	 * 
	 * @param src
	 * @return preference profile provided on PrefLib
	 */
	public static PrefProfile loadPrefProfile(Path src) {
		List<String> lines = null;
		try {
			lines = Files.readAllLines(src);
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}

		int numItem = Integer.parseInt(lines.get(0));

		List<List<Position>> preferences = new ArrayList<>();
		List<Integer> votes = new ArrayList<>();
		List<Integer> items = new ArrayList<>();

		String[] fields = null;
		for (int i = 1; i <= numItem; i++) {
			fields = lines.get(i).split(",");
			items.add(Integer.parseInt(fields[0]));
		}

		for (int i = numItem + 2; i < lines.size(); i++) {
			fields = lines.get(i).split(",");
			votes.add(Integer.parseInt(fields[0]));
			List<Position> preference = new ArrayList<>();

			for (int k = 1, p = 0; k < fields.length; k++, p++) {
				String field = fields[k].trim();
				if (field.startsWith("{")) {
					if (field.equals("{}")) {
						p--;
						continue;
					}

					List<Integer> tied = new ArrayList<>();
					Position position = new Position(p, tied);

					// {4}
					if (field.endsWith("}")) {
						field = field.replace("{", "").replace("}", "");
						tied.add(Integer.parseInt(field));
					} else {
						field = field.replace("{", "");
						tied.add(Integer.parseInt(field));
						k++;
						while (!fields[k].endsWith("}"))
							tied.add(Integer.parseInt(fields[k++].trim()));
						tied.add(Integer.parseInt(fields[k].trim().replace("}", "")));
					}
					position.items = tied;
					preference.add(position);
				} else
					preference.add(new Position(p, Integer.parseInt(field)));
			}
			preferences.add(preference);
		}
		return new PrefProfile(preferences, votes, items);
	}

	/**
	 * Load profile from local file
	 * 
	 * @param path
	 * @return Profile
	 */
	public static Profile<Integer> loadProfile(Path path) {
		List<String> lines = null;
		try {
			lines = Files.readAllLines(path);
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}

		int numItem = Integer.parseInt(lines.get(0));
		int nv = lines.size() - numItem - 2;
		Integer[][] data = new Integer[nv][numItem];
		int[] votes = new int[nv];
		Pattern p = Pattern.compile("\\d+(\\.\\d+)?");
		Matcher m = null;
		for (int i = numItem + 2; i < lines.size(); i++) {
			m = p.matcher(lines.get(i));
			int index = i - numItem - 2;
			if (m.find())
				votes[index] = Integer.parseInt(m.group());
			int k = 0;
			while (m.find()) {
				data[index][k++] = Integer.parseInt(m.group());
			}
		}
		return new Profile<>(data, votes);
	}

	public static PrefProfile getPrefProfile(Profile<Integer> profile) {
		int nRanking = profile.data.length;
		List<List<Position>> preferences = new ArrayList<>(nRanking);
		List<Integer> voteList = new ArrayList<>(nRanking);

		for (int i = 0; i < nRanking; i++) {
			Integer[] ranking = profile.data[i];
			List<Position> pref = new ArrayList<>(ranking.length);
			for (int k = 0; k < ranking.length; k++)
				pref.add(new Position(k, ranking[k]));
			preferences.add(pref);
			voteList.add(profile.votes[i]);
		}
		PrefProfile pp = new PrefProfile(preferences, voteList, Arrays.asList(profile.getSortedItems()));
		return pp;
	}

	/**
	 * Generating single peaked profiles using the recursive approach proposed
	 * by Toby Walsh.
	 * 
	 * @param axis
	 * @return single-peaked preference ranking
	 */
	public static Integer[] getRandomSinglePeakedVote(List<Integer> axis) {
		int sz = axis.size();
		Integer[] vote = new Integer[sz];
		int i1 = 0, i2 = sz - 1, c = sz - 1;
		while (c >= 0) {
			double rand = Math.random();
			if (rand < 0.5) {
				vote[c] = axis.get(i2);
				i2--;
			} else {
				vote[c] = axis.get(i1);
				i1++;
			}
			c--;
		}
		return vote;
	}

	/**
	 * Generate single-peaked profile given m and n in random
	 * 
	 * @param numItem
	 * @param numVote
	 * @return single-peaked profile
	 */
	public static Profile<Integer> getRandomSinglePeakedProfile(int numItem, int numVote) {
		Integer[][] data = new Integer[numVote][];
		for (int i = 0; i < numVote; i++)
			data[i] = getRandomSinglePeakedVote(Arrays.asList(getRandomPermutation(numItem)));

		Profile<Integer> profile = new Profile<>(data);
		profile = profile.compress();
		return profile;

	}

	/**
	 * Generate single-peaked profile given the axis and n in random
	 * 
	 * @param axis
	 * @param numVote
	 * @return single-peaked profile
	 */
	public static Profile<Integer> getRandomSinglePeakedProfile(List<Integer> axis, int numVote) {
		Integer[][] data = new Integer[numVote][];
		for (int i = 0; i < numVote; i++)
			data[i] = getRandomSinglePeakedVote(axis);
		Profile<Integer> profile = new Profile<>(data);
		profile = profile.compress();
		return profile;

	}

	/**
	 * Generate random single-peaked profiles for given m, n and num of samples
	 * 
	 * @param numItem
	 * @param numVote
	 * @param numSample
	 * @param baseFile
	 */
	public static void generateRandomSinglePeakedProfiles(int numItem, int numVote, int numSample, String baseFile) {
		StringBuffer meta = new StringBuffer();
		meta.append(numItem + "\n");
		for (int i = 0; i < numItem; i++)
			meta.append(i + ",c" + i + "\n");

		StringBuffer sb = null;
		Integer[][] data = null;
		String socFile = "";
		OpenOption[] options = { StandardOpenOption.CREATE, StandardOpenOption.WRITE,
				StandardOpenOption.TRUNCATE_EXISTING };

		Profile<Integer> profile = null;
		// fixed axis: 0,1,2,...m
		List<Integer> axis = new ArrayList<>();
		for (int i = 0; i < numItem; i++)
			axis.add(i);

		for (int s = 1; s <= numSample; s++) {
			profile = DataEngine.getRandomSinglePeakedProfile(axis, numVote);
			sb = new StringBuffer();
			sb.append(meta);
			sb.append(numVote + "," + numVote + "," + profile.data.length + "\n");

			data = profile.data;
			for (int i = 0; i < data.length; i++) {
				sb.append(profile.votes[i]);
				for (int k = 0; k < data[i].length; k++)
					sb.append("," + data[i][k]);
				sb.append("\n");
			}

			String name = "M" + numItem + "N" + numVote + "-" + s + ".csv";
			try {
				socFile = baseFile + "/" + name;
				Files.write(Paths.get(socFile), sb.toString().getBytes(), options);
			} catch (IOException e) {
				e.printStackTrace();
				return;
			}
		}
	}

	/***
	 * Generate hard cases
	 * 
	 * @param rule
	 * @param numItem
	 * @param numVote
	 * @param numSample
	 * @param baseFile
	 */
	public static void getHardCases(String rule, int numItem, int numVote, int numSample, String baseFile) {
		StringBuffer meta = new StringBuffer();
		meta.append(numItem + "\n");
		for (int i = 0; i < numItem; i++)
			meta.append(i + ",c" + i + "\n");

		StringBuffer sb = null;
		Integer[][] data = null;
		String socFile = "";
		OpenOption[] options = { StandardOpenOption.CREATE, StandardOpenOption.WRITE,
				StandardOpenOption.TRUNCATE_EXISTING };

		Profile<Integer> profile = null;
		int s = 0;
		while (s < numSample) {
			profile = DataEngine.getRandomProfile(numItem, numVote);
			int hardness = VoteLab.getHardness(rule, profile);
			if (hardness == 1 || (rule.contains("stv") && hardness < 3))
				continue;

			s++;
			sb = new StringBuffer();
			sb.append(meta);
			sb.append(numVote + "," + numVote + "," + profile.data.length + "\n");

			data = profile.data;
			for (int i = 0; i < data.length; i++) {
				sb.append(profile.votes[i]);
				for (int k = 0; k < data[i].length; k++)
					sb.append("," + data[i][k]);
				sb.append("\n");
			}

			String name = "M" + numItem + "N" + numVote + "-" + s + ".csv";
			try {
				socFile = baseFile + "/" + name;
				Files.write(Paths.get(socFile), sb.toString().getBytes(), options);
			} catch (IOException e) {
				e.printStackTrace();
				return;
			}
		}
	}

	/**
	 * Generate single peaked hard cases
	 * 
	 * @param rule
	 * @param numItem
	 * @param numVote
	 * @param numSample
	 * @param baseFile
	 */
	public static void getHardCasesSinglePeaked(String rule, int numItem, int numVote, int numSample, String baseFile) {
		StringBuffer meta = new StringBuffer();
		meta.append(numItem + "\n");
		for (int i = 0; i < numItem; i++)
			meta.append(i + ",c" + i + "\n");

		StringBuffer sb = null;
		Integer[][] data = null;
		String socFile = "";
		OpenOption[] options = { StandardOpenOption.CREATE, StandardOpenOption.WRITE,
				StandardOpenOption.TRUNCATE_EXISTING };

		Profile<Integer> profile = null;
		// fixed axis: 0,1,2,...m
		List<Integer> axis = new ArrayList<>();
		for (int i = 0; i < numItem; i++)
			axis.add(i);

		int s = 0;
		while (s < numSample) {
			profile = DataEngine.getRandomSinglePeakedProfile(axis, numVote);
			int hardness = VoteLab.getHardness(rule, profile);
			if (hardness == 1 || (rule.contains("stv") && hardness < 3))
				continue;

			s++;
			sb = new StringBuffer();
			sb.append(meta);
			sb.append(numVote + "," + numVote + "," + profile.data.length + "\n");

			data = profile.data;
			for (int i = 0; i < data.length; i++) {
				sb.append(profile.votes[i]);
				for (int k = 0; k < data[i].length; k++)
					sb.append("," + data[i][k]);
				sb.append("\n");
			}

			String name = "M" + numItem + "N" + numVote + "-" + s + ".csv";
			try {
				socFile = baseFile + "/" + name;
				Files.write(Paths.get(socFile), sb.toString().getBytes(), options);
			} catch (IOException e) {
				e.printStackTrace();
				return;
			}
		}
	}

	public static void main1111(String[] args) throws Exception {
		TickClock.beginTick();

		List<String> easy = Files.readAllLines(Paths.get("/Users/chjiang/Documents/csc/easy.txt"));
		List<String[]> lines = easy.stream().map(line -> line.split("\t")).collect(Collectors.toList());
		Map<Integer, List<Integer>> map = new HashMap<>();
		for (String[] line : lines) {
			int k = Integer.parseInt(line[0]);
			int v = Integer.parseInt(line[1]);

			List<Integer> value = map.get(k);
			if (value == null)
				value = new ArrayList<>();
			value.add(v);
			map.put(k, value);
		}

		String base = "/Users/chjiang/Documents/csc/";
		CopyOption[] cpOptions = { StandardCopyOption.COPY_ATTRIBUTES, StandardCopyOption.REPLACE_EXISTING };

		Profile<Integer> profile;
		for (int n = 10; n <= 100; n += 10) {
			List<Integer> value = map.get(n);
			if (value == null)
				continue;

			int count = 1;
			while (count <= value.size()) {
				DataEngine.generateRandomProfiles(10, n, 1, base);
				String name = "M10N" + n + "-1.csv";
				Path src = Paths.get(base + name);
				profile = DataEngine.loadProfile(src);
				int h = VoteLab.getHardness("stv", profile);
				if (h == 3) {
					Path dest = Paths.get(base + "/argument/M10N" + n + "-" + (1000 + count) + ".csv");
					Files.copy(src, dest, cpOptions);
					count++;
				}
			}
		}

		TickClock.stopTick();
	}

	public static void main11(String[] args) throws IOException {
		TickClock.beginTick();

		String baseFile = "/Users/chjiang/Documents/csc/";
		String dataset = "soc-33";

		int s = 2000;
		String[] rules = { "coombs" };
		for (String rule : rules) {
			Files.createDirectories(Paths.get(baseFile + dataset + "-" + rule + "/"));
			for (int i = 1; i <= 3; i++) {
				int m = 10 * i;
				for (int j = 1; j <= 10; j++) {
					int n = 10 * j;
					DataEngine.getHardCases(rule, m, n, s, baseFile + dataset + "-" + rule + "/");
					System.out.println(i + ", " + j);
				}
			}
		}
		TickClock.stopTick();
	}

	public static void main0000(String[] args) throws IOException {
		TickClock.beginTick();

		String[] rules = { "baldwin", "coombs", "stv" };
		StringBuffer sb = new StringBuffer();
		sb.append("m");
		for (int n = 10; n <= 100; n += 10)
			sb.append("\t" + n);

		Profile<Integer> profile;
		int maxm = 30, maxn = 100, maxk = 1000;
		for (String rule : rules) {
			System.out.println(sb.toString());
			for (int m = 10; m <= maxm; m += 10) {
				System.out.print(m);
				for (int n = 10; n <= maxn; n += 10) {
					int count = 0;
					for (int k = 0; k < maxk; k++) {
						profile = DataEngine.getRandomSinglePeakedProfile(m, n);
						int hardness = VoteLab.getHardness(rule, profile);
						if (rule.contains("stv")) {
							if (hardness == 3)
								count++;
						} else if (hardness == 2)
							count++;
					}
					System.out.print("\t" + count * 1.0d / maxk);
				}
				System.out.println();
			}
		}

		TickClock.stopTick();
	}
}