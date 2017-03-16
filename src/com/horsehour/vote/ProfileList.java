package com.horsehour.vote;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class ProfileList<T> extends ArrayList<Profile<T>> {

	private static final long serialVersionUID = 2624548982957801856L;

	private String name = null;

	public ProfileList(int numPartitions) {
		super(numPartitions);
	}

	public ProfileList(List<Profile<T>> preferences) {
		super(preferences);
	}

	public void setString(String name){
		this.name = name;
	}

	@Override
	public String toString(){
		return name == null ? super.toString() : name;
	}

	public T[] getCandidates(){
		return get(0).getSortedItems();

	}

	public ProfileList<T> reduceToSubsets(int subsetSize, Random rnd){
		ProfileList<T> shrunken = new ProfileList<>(size());

		for (Profile<T> profile : this) {
			shrunken.add(profile.copyRandomSubset(subsetSize, rnd));
		}

		return shrunken;
	}

	public Profile<T> concatenate(){
		int size = 0;
		for (Profile<T> profile : this) {
			size += profile.data.length;
		}

		@SuppressWarnings("unchecked")
		T[][] arr = (T[][]) Array.newInstance(this.get(0).data.getClass().getComponentType(), size);

		int idx = 0;
		for (Profile<T> profile : this) {
			System.arraycopy(profile.data, 0, arr, idx, profile.data.length);
			idx += profile.data.length;
		}
		return new Profile<T>(arr);
	}

	public static <T> ProfileList<T> singleton(Profile<T> profile){
		return new ProfileList<T>(Collections.singletonList(profile));
	}
}
