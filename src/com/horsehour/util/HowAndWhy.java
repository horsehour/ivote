package com.horsehour.util;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDate;
import java.time.temporal.JulianFields;
import java.time.temporal.TemporalField;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.tuple.Pair;

import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;

public class HowAndWhy {

	/**
	 * Tricks about array, list and stream
	 */
	public static <T> void tricks() {
		/**
		 * [Arrays] construct list from array
		 */
		List<Integer> list1 = Arrays.asList(1, 3, 5, 7, 8, 10);

		/**
		 * [IntStream, Stream] construct a integer range list
		 */
		List<Integer> list2 = IntStream.range(0, 10).boxed().collect(Collectors.toList());

		/**
		 * [toArray] list to array
		 */
		Integer[] array1 = list1.toArray(new Integer[0]);
		Integer[] array2 = (Integer[]) list2.toArray();

		/**
		 * [Reflective] to construct an integer array
		 */
		Integer[] array3 = (Integer[]) Array.newInstance(list2.getClass().getComponentType(), list2.size());

		/**
		 * [Collections] fill all position of a list with a specific value
		 */
		Collections.fill(list1, 9);
		/**
		 * [Arrays] fill all position of an array with a specific value
		 */
		int[] array5 = new int[5];
		Arrays.fill(array5, 1);// primitive
		Arrays.fill(array3, 3);// generic

		Arrays.parallelPrefix(array5, (a, b) -> a + b);// culmulate

		/**
		 * [Arrays] copy to construct an array, convert from array to list
		 */
		Integer[] array4 = Arrays.copyOf(array1, array2.length);// shallow copy
		list2 = Arrays.asList(array4);

		/**
		 * [Arrays, Stream] deep copy
		 */
		Integer[][] array6 = { { 1, 3 } };
		Arrays.stream(array6).map(x -> x.clone()).toArray(Integer[][]::new);

		/**
		 * [ArrayUtils] handles arrays as it's like lists: add, remove, insert
		 * (add), search(contains, indexOf)
		 */
		array5 = ArrayUtils.add(array5, 5);
		array5 = ArrayUtils.add(array5, 1, 8);
		// add an integer array to 2d array
		array6 = ArrayUtils.add(array6, new Integer[3]);
		ArrayUtils.remove(array4, 4);
		ArrayUtils.contains(array5, 0);
		ArrayUtils.indexOf(array3, 8);
		ArrayUtils.isEmpty(array1);

		/**
		 * [IntStream] boxed each element of a primitive array and convert to
		 * list
		 */
		list1 = IntStream.of(array5).boxed().collect(Collectors.toList());
		/**
		 * [Stream] convert list to a primitive array
		 */
		array5 = list1.stream().mapToInt(Integer::valueOf).toArray();
		/**
		 * [IntStream] boxed each element of a primitive array and convert to
		 * generic array
		 */
		array1 = IntStream.of(array5).boxed().toArray(Integer[]::new);

		/**
		 * [Stream] counts elements in the list and returns a map from elements
		 * to its frequency in the list
		 */
		Map<Integer, Long> map1 = list1.stream().collect(Collectors.groupingBy(e -> e, Collectors.counting()));
		array1 = map1.keySet().toArray(new Integer[0]);

		/**
		 * [Stream] divides elements in list to two group, even or odd numbers
		 */
		Map<Integer, List<Integer>> map2 = list1.parallelStream().collect(Collectors.groupingBy(e -> e % 2));
		map2.size();

		/**
		 * [Arrays] show array in terms of string
		 */
		Arrays.toString(array5);
		list2.toString();

		/**
		 * [List] sorts with lambda expression
		 */
		list1.sort((a, b) -> a.compareTo(b));

		/**
		 * [Stream] finds prime and makes count
		 */
		Stream<Integer> permutations = IntStream.rangeClosed(1, 100000).parallel().boxed();
		System.out.println(permutations.filter(HowAndWhy::isPrime).count());

		/**
		 * [Stream, Iterator] constructs stream using an iterator
		 */
		Iterator<Integer> iter = list1.iterator();
		StreamSupport.stream(Spliterators.spliteratorUnknownSize(iter, Spliterator.ORDERED), false);

		/**
		 * [Stream, Iterator] zip two equilength stream
		 */
		Stream<Integer> s1 = list1.stream();
		Stream<Integer> s2 = list2.stream();
		Iterator<Integer> it = s1.iterator();
		Stream<Pair<Integer, Integer>> zip = s2.filter(e -> it.hasNext()).map(e -> Pair.of(e, iter.next()));
		zip.close();

		/**
		 * [Stream] Add an element to a stream
		 */
		s2 = Stream.concat(s1, Stream.of(list1.get(0)));// s1 has been consumed
	}

	/**
	 * Evaluate the prime
	 * 
	 * @param n
	 * @return prime returns true, else return false
	 */
	public static boolean isPrime(int n) {
		if (n <= 1)
			return false;
		else if (n <= 3)
			return true;
		else if (n % 2 == 0 || n % 3 == 0)
			return false;
		int i = 5;
		while (i * i <= n) {
			if (n % i == 0 || n % (i + 2) == 0)
				return false;
			i += 6;
		}
		return true;
	}

	/**
	 * Bitwise operation to partition a list into two sublists
	 * 
	 * @param list
	 * @return
	 */
	public static List<List<Integer>> partitionList(List<Integer> list) {
		int sz = list.size();
		List<List<Integer>> partitions = new ArrayList<>();
		for (int i = 0; i < 1 << sz; i++) {
			List<Integer> inclusive = new ArrayList<>();
			for (int j = 0; j < sz; j++) {
				if ((i & 1 << j) > 0)
					inclusive.add(list.get(j));
			}
			partitions.add(inclusive);
		}
		return partitions;
	}

	public static Map<Long, Integer> mergeMap(Map<Long, Integer> map1, Map<Long, Integer> map2) {
		return Stream.concat(map1.entrySet().stream(), map2.entrySet().stream())
				.collect(Collectors.toMap(entry -> entry.getKey(), entry -> entry.getValue(), Integer::sum));
	}

	/**
	 * @param year
	 * @param month
	 * @param day
	 * @return google date range searching
	 */
	public static String googleDateRange(int year, int month, int day) {
		LocalDate dateBegin = LocalDate.of(year, month, day);
		LocalDate dateEnd = LocalDate.now();
		TemporalField field = JulianFields.JULIAN_DAY;
		return dateBegin.getLong(field) + "-" + dateEnd.getLong(field);
	}

	public static int[] split(int m) {
		int n = (int) Math.sqrt(m) + 1;
		System.out.println(n);
		int index = 0, minDiff = 0;
		for (int i = -1; i <= 1; i++) {
			int k = m / (n + i);
			if (k * (n + i) >= m) {
				int diff = Math.abs(k - n - i);
				if (diff < minDiff) {
					minDiff = diff;
					index = i;
				}
			}
		}

		int[] ret = { n + index, m / (n + index) };
		Arrays.sort(ret);
		return ret;
	}

	public static Pair<double[][], int[]> getData(SampleSet sampleset) {
		double[][] data = new double[sampleset.size()][sampleset.dim()];
		int[] labels = new int[sampleset.size()];

		for (int i = 0; i < sampleset.size(); i++) {
			Sample sample = sampleset.getSample(i);
			for (int j = 0; j < sampleset.dim(); j++) {
				data[i][j] = sample.getFeature(j);
				labels[j] = sample.getLabel() - 1;
			}
		}
		return Pair.of(data, labels);
	}

	public static void writeListToFile(List<String> list, String dest) {
		OpenOption[] options = { StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.APPEND };
		try {
			Files.write(Paths.get(dest), list, options);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static List<String> readLines(String src) {
		List<String> lines = null;
		try {
			lines = Files.readAllLines(Paths.get(src));
		} catch (IOException e) {
			e.printStackTrace();
		}
		return lines;
	}

	public static void listAllFiles(String dir) {
		Stream<Path> stream = null;
		try {
			stream = Files.list(Paths.get(dir));
		} catch (IOException e) {
			e.printStackTrace();
		}
		stream.forEach(System.out::println);
	}

	public static void listAllFiles(Path dir) throws IOException {
		DirectoryStream<Path> stream = Files.newDirectoryStream(dir);
		stream.forEach(System.out::println);
	}

	public static void zip(Path dir, Path zipFile) throws IOException {
		ZipOutputStream zipStream = new ZipOutputStream(new FileOutputStream(zipFile.toFile()));
		DirectoryStream<Path> streamDir = Files.newDirectoryStream(dir);
		streamDir.forEach(path -> zip(path.toFile(), zipStream));
		zipStream.close();
	}

	public static void zip(File file, ZipOutputStream zipStream) {
		try (FileInputStream inputStream = new FileInputStream(file)) {
			ZipEntry entry = new ZipEntry(file.getName());
			zipStream.putNextEntry(entry);

			byte[] buffer = new byte[2048];
			int size;
			while ((size = inputStream.read(buffer)) > 0) {
				zipStream.write(buffer, 0, size);
			}

		} catch (IOException e) {
			System.err.println();
		}
	}
	
	public static void initListFixedSize(){
		List<Integer> list = Collections.nCopies(5, 1);
		System.out.println(list);
	}
	
	public static void main(String[] args){
		List<Integer> list = new ArrayList<>(Collections.nCopies(5, 1));
		System.out.println(list);
		list.set(0, 2);
		System.out.println(list);
	}
}
