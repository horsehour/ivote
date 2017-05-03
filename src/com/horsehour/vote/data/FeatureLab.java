package com.horsehour.vote.data;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

import org.apache.commons.lang3.ArrayUtils;

import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
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

/**
 * Laboratory on feature extractions from preference profiles. The features that
 * we extracted from preference profiles have many categories. The dimension of
 * the extracted features may rely on the relative order of alternatives, the
 * number of voters, the number of alternatives, and various general scoring
 * rules (plurality score, Borda score, veto score), even the weighted majority
 * graph.
 * 
 * @author Chunheng Jiang
 * @since Mar 26, 2017
 * @version 1.0
 */
public class FeatureLab {
	public static enum Family {
		F1, F2, F3
	}

	/**
	 * Extracted from profile features, including normalized positional
	 * features, normalized weighted majority scores and normalized Copeland
	 * scores
	 * 
	 * @param profile
	 * @param inclusive
	 *            included alternatives
	 * @param m
	 *            complete number of alternatives
	 * @return normalized positional and pairwise features
	 */
	public static double[] getF2(Profile<Integer> profile, List<Integer> inclusive, int m) {
		int n = profile.numVoteTotal;
		int[][] ppm = new int[m][m];
		int[][] positional = new int[m][m];

		int size = inclusive.size();
		for (int k = 0; k < profile.data.length; k++) {
			Integer[] preference = profile.data[k];
			int top = 0;
			for (int i = 0; i < preference.length && top < size; i++) {
				int ind = inclusive.indexOf(preference[i]);
				if (ind == -1) {
					for (int item : inclusive)
						ppm[item][preference[i]] = n;
					continue;
				}

				// included
				positional[preference[i]][top] += profile.votes[k];
				top++;

				for (int j = i + 1; j < preference.length; j++)
					ppm[preference[i]][preference[j]] += profile.votes[k];
			}
		}

		List<Double> features = new ArrayList<>();
		double max = -1;

		double[] win = new double[m];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				// normalized positional features
				features.add(positional[i][j] * 1.0 / n);

				if (j <= i)
					continue;
				if (ppm[i][j] > ppm[j][i]) {
					ppm[i][j] -= ppm[j][i];
					ppm[j][i] = -ppm[i][j];
					max = Math.max(max, ppm[i][j]);
				} else if (ppm[i][j] < ppm[j][i]) {
					ppm[j][i] -= ppm[i][j];
					ppm[i][j] = -ppm[j][i];
					max = Math.max(max, ppm[j][i]);
				} else {
					ppm[i][j] = ppm[j][i] = 0;
				}
			}
		}

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				if (j == i)
					continue;

				// normalized weighted majority score
				features.add(ppm[i][j] * 1.0 / max);

				if (ppm[i][j] > 0)
					win[i]++;
				else if (ppm[i][j] == 0)
					win[i] += 0.5;
			}
		}

		// normalized Copeland score
		for (int i = 0; i < m; i++)
			features.add(win[i] / (m - 1));

		int d = features.size();
		double[] values = new double[d];
		for (int i = 0; i < d; i++)
			values[i] = features.get(i);
		return values;
	}

	/**
	 * Extracted from profile features, including normalized positional
	 * features, normalized weighted majority scores and normalized Copeland
	 * scores
	 * 
	 * @param profile
	 * @param inclusive
	 *            included alternatives
	 * @param m
	 *            complete number of alternatives
	 * @return normalized positional and pairwise features
	 */
	public static double[] getF1(Profile<Integer> profile, List<Integer> inclusive, int m) {
		int n = profile.numVoteTotal;
		int[][] ppm = new int[m][m];
		int[][] positional = new int[m][m];

		int size = inclusive.size();
		for (int k = 0; k < profile.data.length; k++) {
			Integer[] preference = profile.data[k];
			int top = 0;
			for (int i = 0; i < preference.length && top < size; i++) {
				int ind = inclusive.indexOf(preference[i]);
				if (ind == -1) {
					for (int item : inclusive)
						ppm[item][preference[i]] = n;
					continue;
				}

				// included
				positional[preference[i]][top] += profile.votes[k];
				top++;

				for (int j = i + 1; j < preference.length; j++)
					ppm[preference[i]][preference[j]] += profile.votes[k];
			}
		}

		List<Double> features = new ArrayList<>();
		double max = -1;

		double[] win = new double[m];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				// normalized positional features
				features.add(positional[i][j] * 1.0 / n);

				if (j <= i)
					continue;
				if (ppm[i][j] > ppm[j][i]) {
					ppm[i][j] -= ppm[j][i];
					ppm[j][i] = -ppm[i][j];
					max = Math.max(max, ppm[i][j]);
				} else if (ppm[i][j] < ppm[j][i]) {
					ppm[j][i] -= ppm[i][j];
					ppm[i][j] = -ppm[j][i];
					max = Math.max(max, ppm[j][i]);
				} else {
					ppm[i][j] = ppm[j][i] = 0;
				}
			}
		}

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				if (j == i)
					continue;

				// normalized weighted majority score
				// TODO
				if (j > i)
					features.add(ppm[i][j] * 1.0 / max);

				if (ppm[i][j] > 0)
					win[i]++;
				else if (ppm[i][j] == 0)
					win[i] += 0.5;
			}
		}

		// normalized Copeland score
		for (int i = 0; i < m; i++)
			features.add(win[i] / (m - 1));

		int d = features.size();
		double[] values = new double[d];
		for (int i = 0; i < d; i++)
			values[i] = features.get(i);
		return values;
	}

	public static double[] getF9(Profile<Integer> profile, List<Integer> items) {
		int m = items.size(), n = profile.numVoteTotal;
		int dim = n * (m * (m - 1)) / 2, c = 0;
		double[] features = new double[dim];
		for (Integer[] preference : profile.data) {
			List<Integer> indices = new ArrayList<>();
			for (int pos : preference) {
				int ind = items.indexOf(pos);
				if (ind > -1)
					indices.add(ind);
			}

			int[][] matrix = new int[m][m];
			for (int i = 0; i < m; i++) {
				int a = indices.get(i);
				for (int j = i + 1; j < m; j++) {
					int b = indices.get(j);
					matrix[a][b] = 1;
				}
			}

			for (int i = 0; i < m; i++)
				for (int j = i + 1; j < m; j++)
					features[c++] = matrix[i][j];
		}
		return features;
	}

	public static double[] getF2(Profile<Integer> profile, List<Integer> inclusive) {
		int m = inclusive.size();
		int numPreferences = profile.data.length;
		int[][] positions = new int[m][numPreferences];

		for (int i = 0; i < m; i++) {
			for (int k = 0; k < numPreferences; k++) {
				Integer[] preferences = profile.data[k];
				positions[i][k] = ArrayUtils.indexOf(preferences, inclusive.get(i));
			}
		}

		int nv = profile.numVoteTotal;
		List<Float> features = new ArrayList<>();
		for (int i = 0; i < m; i++) {
			for (int j = i + 1; j < m; j++) {
				float[][] pairFeature = new float[m][m];
				for (int k = 0; k < numPreferences; k++) {
					int posI = positions[i][k];
					int posJ = positions[j][k];

					pairFeature[posI][posJ] += (profile.votes[k] * 1.0F / nv);
				}
				for (int idxI = 0; idxI < m; idxI++)
					for (int idxJ = 0; idxJ < m; idxJ++) {
						if (idxI == idxJ)
							continue;
						features.add(pairFeature[idxI][idxJ]);
					}
			}
		}

		double[] results = new double[features.size()];
		for (int i = 0; i < features.size(); i++)
			results[i] = features.get(i);
		return results;
	}

	/**
	 * @param profile
	 * @return Features extracted from profile
	 */
	public static List<Double> getFeatures(Profile<Integer> profile) {
		Integer[] items = profile.getSortedItems();

		List<Double> features = new ArrayList<>();
		features.addAll(getFeatures(profile, items));
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
		List<List<T>> permutations = DataEngine.getAllPermutations(Arrays.asList(items));
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
		List<List<T>> permutations = DataEngine.getAllPermutations(Arrays.asList(items));

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

	public static List<Float> getPositionalFeatures(Profile<Integer> profile, Integer[] items) {
		int numItem = items.length;
		int numVote = profile.getNumVote();

		int[][] positionalFeatures = new int[numItem][numItem];

		int k = 0;
		for (Integer[] pref : profile.data) {
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
	public static <T> List<Float> getPairwiseFeatures(Profile<T> profile, T[] items) {
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
	public static <T> List<float[][]> getGeneralFeatures(Profile<Integer> profile, Integer[] items) {
		if (items == null)
			items = profile.getSortedItems();

		int numItem = items.length;
		int numPreferences = profile.data.length;
		int[][] indices = new int[numItem][numPreferences];

		for (int i = 0; i < numItem; i++)
			for (int k = 0; k < numPreferences; k++) {
				Integer[] preferences = profile.data[k];
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

	/**
	 * @param profile
	 * @param items
	 * @param inclusive
	 *            true if items are used to indicate the alternatives should be
	 *            included in the profile, false if they are exclusive from the
	 *            profile
	 * @return features
	 */
	public static double[] getFeatures(Profile<Integer> profile, List<Integer> items, boolean inclusive) {

		return null;
	}

	public static List<Double> getFeatures(Profile<Integer> profile, Integer[] items) {
		if (items == null)
			items = profile.getSortedItems();

		int m = items.length;
		int numPreferences = profile.data.length;
		int[][] positions = new int[m][numPreferences];

		for (int i = 0; i < m; i++) {
			for (int k = 0; k < numPreferences; k++) {
				Integer[] preferences = profile.data[k];
				int ind = ArrayUtils.indexOf(preferences, items[i]);
				if (ind > -1)
					positions[i][k] = ind;
			}
		}

		int n = profile.numVoteTotal;
		List<Double> features = new ArrayList<>();
		for (int i = 0; i < m; i++) {
			for (int j = i + 1; j < m; j++) {
				double[][] pairFeature = new double[m][m];
				for (int k = 0; k < numPreferences; k++) {
					int posI = positions[i][k];
					int posJ = positions[j][k];
					pairFeature[posI][posJ] += (profile.votes[k] * 1.0F / n);
				}
				for (int idxI = 0; idxI < m; idxI++)
					for (int idxJ = 0; idxJ < m; idxJ++) {
						if (idxI == idxJ)
							continue;
						else
							features.add(pairFeature[idxI][idxJ]);
					}
			}
		}
		return features;
	}

	/**
	 * Features including normalized Borda score, plurality score, veto score
	 * and normalized winning rate are extracted for all individual alternatives
	 * 
	 * @param profile
	 * @param items
	 * @return individual features for all alternatives
	 */
	public static double[][] getFeatures(Profile<Integer> profile, List<Integer> items) {
		int m = items.size();
		int n = profile.numVoteTotal;
		int numPreferences = profile.data.length;
		int[][] positions = new int[m][numPreferences];

		double[][] features = new double[m][4];
		double bordaSUM = 0;

		for (int i = 0; i < numPreferences; i++) {
			Integer[] preferences = profile.data[i];
			int p = 0;
			for (int j = 0; j < preferences.length; j++) {
				int ind = -1;
				if ((ind = items.indexOf(preferences[j])) > -1) {
					p++;
					// Borda score
					features[ind][0] += profile.votes[i] * (m - p);
					// normalized plurality score
					features[ind][1] += (p == 1 ? (profile.votes[i] * 1.0F / n) : 0);
					// normalized (negative) veto score
					features[ind][2] -= (p == m ? (profile.votes[i] * 1.0F / n) : 0);
					bordaSUM += features[ind][0];
					positions[ind][i] = p;
				}
			}
		}

		// normalized winning rate
		for (int i = 0; i < m; i++) {
			// normalized Borda score
			features[i][0] /= bordaSUM;
			for (int j = i + 1; j < m; j++) {
				for (int k = 0; k < numPreferences; k++) {
					float weight = (profile.votes[k] * 1.0F / n) / (m - 1);
					if (positions[i][k] < positions[j][k])
						features[i][3] += weight;
					else
						features[j][3] += weight;
				}
			}
		}
		return features;
	}

	public static Function<String[], int[]> decode = bin -> {
		int dim = bin.length;
		List<Integer> list = new ArrayList<>();
		for (int i = 0; i < dim; i++)
			if (bin[i].contains("1"))
				list.add(i);

		int[] cls = new int[list.size()];
		for (int i = 0; i < list.size(); i++)
			cls[i] = list.get(i);
		return cls;
	};

	/**
	 * K Hot Decoding
	 */
	public static Function<String, List<Integer>> hotDecode = binary -> {
		List<Integer> winners = new ArrayList<>();
		String[] b = binary.split(",");
		for (int i = 0; i < b.length; i++) {
			if (b[i].contains("1"))
				winners.add(i);
		}
		return winners;
	};

	/**
	 * K Hot Encoding
	 */
	public static BiFunction<Integer, int[], String> hotEncode = (k, output) -> {
		List<Integer> list = new ArrayList<>();
		for (int i : output)
			list.add(i);
		Collections.sort(list);

		StringBuffer sb = new StringBuffer();

		for (int i = 0; i < k - 1; i++) {
			if (!list.isEmpty() && i == list.get(0))
				sb.append("1,");
			else
				sb.append("0,");
		}

		if (!list.isEmpty() && list.get(0) == k - 1)
			sb.append("1");
		else
			sb.append("0");
		return sb.toString();
	};

	public static void main(String[] args) {
		TickClock.beginTick();

		String base = "/Users/chjiang/GitHub/csc/";
		Path file = Paths.get(base + "soc-3/M10N10-1.csv");
		Profile<Integer> profile = DataEngine.loadProfile(file);
		List<Integer> inclusive = new ArrayList<>();
		for (int i = 0; i < 2; i++)
			inclusive.add(i);

		double[] features = FeatureLab.getF1(profile, inclusive, 30);
		System.out.println(features.length);

		TickClock.stopTick();
	}

	public static void main2(String[] args) throws IOException {
		TickClock.beginTick();

		String base = "/Users/chjiang/GitHub/csc/";
		Path file = Paths.get(base + "winners-stv-soc3.txt");
		Path sink = Paths.get(base + "M10-30.csv");

		int m0 = 30;

		Profile<Integer> profile = null;
		List<Integer> inclusive = new ArrayList<>();
		for (int i = 0; i < m0; i++)
			inclusive.add(i);
		OpenOption[] options = { StandardOpenOption.APPEND, StandardOpenOption.CREATE, StandardOpenOption.WRITE };

		StringBuffer sb = null;
		List<String> lines = Files.readAllLines(file);

		for (String line : lines) {
			int ind = line.indexOf("\t");
			String name = line.substring(0, ind);
			file = Paths.get(base + "soc-3-hardcase/" + name);
			int m = Integer.parseInt(name.substring(1, 3));
			if (!Files.exists(file) || m > 30)
				continue;

			profile = DataEngine.loadProfile(file);
			String winners = line.substring(ind + 1).trim();
			sb = new StringBuffer();
			double[] features = FeatureLab.getF2(profile, inclusive, m0);
			String input = Arrays.toString(features).replaceAll("\\[|\\]| ", "");
			sb.append(input).append(";").append(winners);

			for (int i = m; i < 30; i++) {
				sb.append(",").append("0");
			}

			sb.append("\n");
			Files.write(sink, sb.toString().getBytes(), options);
			System.out.println(name);
		}
		TickClock.stopTick();
	}
}
