package com.horsehour.vote.stat;

import jnr.ffi.byref.DoubleByReference;
import jnr.ffi.byref.IntByReference;

/**
 * Direct-mapped library to mvnpack.so This class itself is NOT thread-safe
 * 
 * @author mao
 *
 */
public class MvnPackDirect implements MvnPackGenz {

//	static {
//		Native.register(MvnPackDirect.class.getClassLoader().getResource(MVNPACK_SO).getPath());
//	}

	@Override
	public native void mvndst_(IntByReference n, double[] lower, double[] upper, int[] infin, double[] correl,
			IntByReference maxpts, DoubleByReference abseps, DoubleByReference releps, DoubleByReference error,
			DoubleByReference value, IntByReference inform);

	@Override
	public native void mvnexp_(IntByReference n, double[] lower, double[] upper, int[] infin, double[] correl,
			IntByReference maxpts, DoubleByReference abseps, DoubleByReference releps, double[] error, double[] value,
			IntByReference inform);

	@Override
	public native void mvnxpp_(IntByReference n, double[] lower, double[] upper, int[] infin, double[] correl,
			IntByReference maxpts, DoubleByReference abseps, DoubleByReference releps, double[] error, double[] value,
			IntByReference inform);
}
