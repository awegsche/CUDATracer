#pragma once

#include "constants.h"
#include "QtGui/qimage.h"
#include "RenderObjects/render_structs.h"

float4 Qrgb_to_float4(const QRgb col);

__device__ inline float4 rgbcolor(const float r, const float g, const float b) {
	float4 f;
	f.x = r;
	f.y = g;
	f.z = b;
	f.w = 1.0f;
	return f;
}

__device__ inline float4 rgbcolor(const float r, const float g, const float b, const float alpha) {
	float4 f;
	f.x = r;
	f.y = g;
	f.z = b;
	f.w = alpha;
	return f;
}

__device__ inline float4 operator-(const float4 &a, const float4 &b) {
	return rgbcolor(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__device__ inline float4 rgbcolor(const float brightness) {
	float4 f;
	f.x = brightness;
	f.y = brightness;
	f.z = brightness;
	f.w = 1.0f;
	return f;
}

__device__ inline uchar4 _rgbcolor_to_byte(const float4 &col) {
	uchar4 f;

	float r = clamp(col.x, .0f, 1.f);
	float g = clamp(col.y, .0f, 1.f);
	float b = clamp(col.z, .0f, 1.f);

	f.x = (uchar)(r * 255.0f);
	f.y = (uchar)(g * 255.0f);
	f.z = (uchar)(b * 255.0f);
	f.w = (uchar)(col.w * 255.0f);

	return f;
}

__device__ inline float4 scale_color(const float4 &color, const float a) {
	return rgbcolor(color.x * a, color.y * a, color.z * a, color.w);
}

__device__ inline float4 add_colors(const float4 &a, const float4 &b) {
	return rgbcolor(a.x + b.x, a.y + b.y, a.z + b.z, a.w);
}
