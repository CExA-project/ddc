# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

import numpy as np


def gen_value(dtype):
    base_value = np.complex128(2.3, 0.4)
    if np.issubdtype(dtype, np.complexfloating):
        return dtype(base_value)
    if np.issubdtype(dtype, np.floating):
        return dtype(np.real(base_value))
    return dtype(np.round(np.real(base_value)))


def gen_array_0d(dtype):
    return np.full(shape=(), fill_value=gen_value(dtype), order="C")


def gen_array_1d(dtype):
    return np.full(shape=(2 * 3 * 4), fill_value=gen_value(dtype), order="C")


def gen_array_3d(dtype):
    return np.full(shape=(2, 3, 4), fill_value=gen_value(dtype), order="C")


def test_0d():
    np.testing.assert_array_equal(np.load("float_0d.npy"), gen_array_0d(np.float32), strict=True)


def test_1d():
    np.testing.assert_array_equal(np.load("double_1d.npy"), gen_array_1d(np.float64), strict=True)


def test_3d():
    np.testing.assert_array_equal(np.load("int8_3d.npy"), gen_array_3d(np.int8), strict=True)
    np.testing.assert_array_equal(np.load("int16_3d.npy"), gen_array_3d(np.int16), strict=True)
    np.testing.assert_array_equal(np.load("int32_3d.npy"), gen_array_3d(np.int32), strict=True)
    np.testing.assert_array_equal(np.load("int64_3d.npy"), gen_array_3d(np.int64), strict=True)

    np.testing.assert_array_equal(np.load("uint8_3d.npy"), gen_array_3d(np.uint8), strict=True)
    np.testing.assert_array_equal(np.load("uint16_3d.npy"), gen_array_3d(np.uint16), strict=True)
    np.testing.assert_array_equal(np.load("uint32_3d.npy"), gen_array_3d(np.uint32), strict=True)
    np.testing.assert_array_equal(np.load("uint64_3d.npy"), gen_array_3d(np.uint64), strict=True)

    np.testing.assert_array_equal(np.load("float_3d.npy"), gen_array_3d(np.float32), strict=True)
    np.testing.assert_array_equal(np.load("double_3d.npy"), gen_array_3d(np.float64), strict=True)

    np.testing.assert_array_equal(
        np.load("complex_float_3d.npy"), gen_array_3d(np.complex64), strict=True
    )
    np.testing.assert_array_equal(
        np.load("complex_double_3d.npy"), gen_array_3d(np.complex128), strict=True
    )
