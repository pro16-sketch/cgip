import os
from io import BytesIO
from flask import Flask, request, jsonify, send_file, send_from_directory
import cv2
import numpy as np


app = Flask(__name__, static_folder='.', static_url_path='')


def read_image_file(file_storage):
	data = file_storage.read()
	arr = np.frombuffer(data, np.uint8)
	img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
	if img is None:
		return None
	# Ensure 3-channel BGR
	if len(img.shape) == 2:
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	elif img.shape[2] == 4:
		img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
	return img


def cv2_to_bytes(img, fmt='jpg', quality=90):
	f = fmt.lower()
	if f in ('jpg', 'jpeg'):
		ext = '.jpg'
	elif f == 'png':
		ext = '.png'
	elif f == 'webp':
		ext = '.webp'
	elif f in ('tif', 'tiff'):
		ext = '.tiff'
	else:
		# fallback to jpg
		ext = '.jpg'
	# Build params as a list of ints expected by cv2.imencode
	if ext == '.jpg':
		params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
	else:
		# map quality (1-100) to png compression (0-9)
		comp = max(0, min(9, 9 - int(quality / 11)))
		params = [cv2.IMWRITE_PNG_COMPRESSION, int(comp)]

	# cv2.imencode expects the ext (with dot) and params as list
	success, encoded = cv2.imencode(ext, img, params)
	if not success:
		raise RuntimeError('Failed to encode image')
	return encoded.tobytes()


def adjust_brightness_contrast(img, alpha=1.0, beta=0.0):
	out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
	return out


def to_grayscale(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def blur_image(img, ksize=5):
	k = ksize if ksize % 2 == 1 else ksize + 1
	return cv2.GaussianBlur(img, (k, k), 0)


def sharpen_image(img):
	kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
	return cv2.filter2D(img, -1, kernel)


def edge_detection(img, method='canny'):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	if method == 'sobel':
		gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
		gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
		mag = cv2.magnitude(gx, gy)
		_, out = cv2.threshold(mag, 50, 255, cv2.THRESH_BINARY)
		return out.astype(np.uint8)
	# default: canny
	return cv2.Canny(gray, 100, 200)


def color_filter(img, typ='sepia'):
	# ensure float for transforms
	typ = (typ or 'sepia').lower()
	if typ == 'sepia':
		kernel = np.array([[0.272, 0.534, 0.131],
						   [0.349, 0.686, 0.168],
						   [0.393, 0.769, 0.189]])
		out = cv2.transform(img, kernel)
		out = np.clip(out, 0, 255).astype(np.uint8)
		return out

	if typ == 'negative':
		return cv2.bitwise_not(img)

	if typ == 'emboss':
		kernel = np.array([[ -2, -1, 0],
						   [ -1, 1, 1],
						   [ 0, 1, 2]])
		out = cv2.filter2D(img, -1, kernel) + 128
		return np.clip(out, 0, 255).astype(np.uint8)

	if typ == 'sketch':
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		inv = 255 - gray
		blur = cv2.GaussianBlur(inv, (21, 21), 0)
		sketch = cv2.divide(gray, 255 - blur, scale=256)
		return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

	if typ == 'warm':
		# increase red, reduce blue slightly
		b, g, r = cv2.split(img)
		r = cv2.add(r, 30)
		b = cv2.subtract(b, 10)
		out = cv2.merge([b, g, r])
		return np.clip(out, 0, 255).astype(np.uint8)

	if typ == 'cool':
		# increase blue, reduce red slightly
		b, g, r = cv2.split(img)
		b = cv2.add(b, 30)
		r = cv2.subtract(r, 10)
		out = cv2.merge([b, g, r])
		return np.clip(out, 0, 255).astype(np.uint8)

	if typ == 'vintage':
		# simple vintage: lower saturation and add a warm overlay
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
		hsv[...,1] *= 0.6
		low_sat = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
		overlay = np.full_like(low_sat, (20, 30, 60))
		out = cv2.addWeighted(low_sat, 0.85, overlay, 0.15, 0)
		return np.clip(out, 0, 255).astype(np.uint8)

	# unknown type: return original
	return img


def undo_color_filter(img, typ='sepia'):
	"""Attempt to reverse simple color filters. This is heuristic and best-effort.
	For some filters (negative, sepia, warm, cool, vintage) we can approximate an inverse.
	For sketch/emboss/edge we cannot reliably reconstruct the original, so we return the input.
	"""
	t = (typ or 'sepia').lower()
	if t == 'negative':
		# negative of negative -> original
		return cv2.bitwise_not(img)

	if t == 'sepia':
		# apply pseudo-inverse of sepia transform
		kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
		try:
			inv = np.linalg.pinv(kernel)
			f = img.astype(np.float32)
			# cv2.transform expects shape (h,w,3)
			out = cv2.transform(f, inv)
			out = np.clip(out, 0, 255).astype(np.uint8)
			return out
		except Exception:
			return img

	if t == 'warm':
		# reverse by subtracting added red and adding blue
		b, g, r = cv2.split(img.astype(np.int16))
		r = r - 30
		b = b + 10
		out = cv2.merge([b, g, r]).clip(0, 255).astype(np.uint8)
		return out

	if t == 'cool':
		# reverse cool: subtract blue and add red
		b, g, r = cv2.split(img.astype(np.int16))
		b = b - 30
		r = r + 10
		out = cv2.merge([b, g, r]).clip(0, 255).astype(np.uint8)
		return out

	if t == 'vintage':
		# attempt to restore saturation lost by vintage filter
		try:
			hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
			h, s, v = cv2.split(hsv)
			s = s / 0.6
			s = np.clip(s, 0, 255)
			hsv2 = cv2.merge([h, s, v]).astype(np.uint8)
			out = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
			return out
		except Exception:
			return img

	# emboss, sketch, edge and other structural filters are not reversible
	return img


def rotate_image(img, angle=0.0):
	(h, w) = img.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	return cv2.warpAffine(img, M, (w, h))


def resize_image(img, w, h):
	return cv2.resize(img, (w, h))


def crop_image(img, x, y, w, h):
	return img[y:y+h, x:x+w]


def compress_to_target(img, fmt='jpg', target_bytes=100_000, min_quality=20):
	"""
	Try to compress `img` so encoded size <= target_bytes.
	Strategy:
	1. If fmt is PNG, first try converting to JPEG since JPEG compresses better for photos.
	2. Iteratively reduce JPEG quality from 95 down to min_quality.
	3. If not enough, downscale image by 90% and repeat (up to a few iterations).
	Returns bytes and final mimetype fmt.
	"""
	# prefer jpeg for compression unless user explicitly asked png
	out_fmt = fmt.lower()
	if out_fmt == 'png':
		# try jpeg instead for better size reduction on photos
		out_fmt = 'jpg'

	h, w = img.shape[:2]
	quality = 95
	attempts = 0
	max_downscale_iters = 6

	current = img.copy()
	while attempts < max_downscale_iters:
		q = quality
		while q >= min_quality:
			data = cv2_to_bytes(current, fmt=out_fmt, quality=q)
			if len(data) <= target_bytes or q == min_quality:
				return data, out_fmt
			q -= 5
		# not small enough, downscale
		w = int(w * 0.9)
		h = int(h * 0.9)
		if w < 16 or h < 16:
			break
		current = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
		attempts += 1

	# fallback: return last attempt
	return cv2_to_bytes(current, fmt=out_fmt, quality=min_quality), out_fmt


def cartoonify_image(img, downscale=1, num_bilateral=7, edge_ksize=5):
	"""
	Produces a cartoon-like effect:
	- Apply bilateral filter multiple times to smooth colors while preserving edges
	- Convert to grayscale and detect edges with median blur + adaptive threshold
	- Combine smoothed color image with edges mask
	"""
	# optionally downscale for speed
	if downscale > 1:
		small = cv2.resize(img, (img.shape[1]//downscale, img.shape[0]//downscale), interpolation=cv2.INTER_AREA)
	else:
		small = img.copy()

	# repeated bilateral filtering
	for _ in range(num_bilateral):
		small = cv2.bilateralFilter(small, d=9, sigmaColor=75, sigmaSpace=75)

	# edge mask
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# sanitize edge_ksize: must be odd and >=3 and <= min(image_dim)
	try:
		k = int(edge_ksize)
	except Exception:
		k = 5
	# clamp to image dimensions
	h_img, w_img = gray.shape[:2]
	maxk = min(h_img, w_img)
	if maxk < 3:
		k = 1
	else:
		if k < 3:
			k = 3
		if k > maxk:
			k = maxk if (maxk % 2 == 1) else maxk - 1
		if k % 2 == 0:
			k += 1
			if k > maxk:
				k -= 2
				if k < 3:
					k = 3
	# medianBlur requires odd ksize >=1
	k = max(1, k)
	if k == 1:
		blurred_for_edges = gray
	else:
		blurred_for_edges = cv2.medianBlur(gray, k)
	edges = cv2.adaptiveThreshold(blurred_for_edges, 255,
								  cv2.ADAPTIVE_THRESH_MEAN_C,
								  cv2.THRESH_BINARY,
								  blockSize=9, C=2)

	# combine color with edges
	# if we downscaled, upscale smoothed color to original size
	if small.shape[:2] != img.shape[:2]:
		color = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
	else:
		color = small

	# convert edges to color and bitwise-and with color image
	edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
	cartoon = cv2.bitwise_and(color, edges_color)
	return cartoon


def add_haze(img, intensity=0.5, color_hex='#ffffff', vertical=True):
	"""Add a haze/fog overlay to the image.

	intensity: float 0..1 where 0 is no haze and 1 is full haze.
	color_hex: hex string for haze color (default white).
	vertical: if True, stronger haze toward top (simulates distance)
	"""
	try:
		hval = float(intensity)
	except Exception:
		hval = 0.5
	hval = max(0.0, min(1.0, hval))

	# parse hex color to BGR
	col = (255, 255, 255)
	if color_hex:
		s = str(color_hex).lstrip('#')
		if len(s) == 6:
			try:
				r = int(s[0:2], 16)
				g = int(s[2:4], 16)
				b = int(s[4:6], 16)
				col = (b, g, r)
			except Exception:
				pass

	rows, cols = img.shape[:2]

	# generate smooth noise as haze map
	noise = np.abs(np.random.randn(rows, cols).astype(np.float32))
	noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)

	# blur to make larger soft patches
	k = max(1, int(min(rows, cols) / 20))
	if k % 2 == 0:
		k += 1
	try:
		noise = cv2.GaussianBlur(noise, (k, k), 0)
	except Exception:
		pass

	# gradient to simulate stronger haze at distance (top)
	if vertical:
		grad = np.linspace(1.0, 0.2, rows).reshape(rows, 1).astype(np.float32)
	else:
		grad = np.ones((rows, 1), dtype=np.float32)

	alpha = noise * grad * hval
	alpha = np.clip(alpha, 0.0, 1.0)
	alpha3 = np.dstack([alpha, alpha, alpha])

	color_layer = np.zeros_like(img, dtype=np.float32)
	color_layer[..., 0] = col[0]
	color_layer[..., 1] = col[1]
	color_layer[..., 2] = col[2]

	img_f = img.astype(np.float32)
	out = img_f * (1.0 - alpha3) + color_layer * alpha3
	out = np.clip(out, 0, 255).astype(np.uint8)
	return out


def enhance_image(img, fix_negative=False, denoise_h=0.0, clahe_clip=2.0, sharpen_amount=0.8, gamma=1.0, awb=False):
	"""
	Simple enhancement pipeline intended to brighten and restore old/rusty/blurred/negatived photos.
	Steps (approx):
	- Optionally invert negative images
	- Denoise (fastNlMeans)
	- Auto white balance (gray-world) when requested
	- CLAHE on L channel to improve local contrast
	- Unsharp mask (sharpen)
	- Gamma correction
	"""
	out = img.copy()

	# If the image is negative (user asks), invert it first
	if fix_negative:
		out = cv2.bitwise_not(out)

	# Denoise colored image
	try:
		h = float(denoise_h)
	except Exception:
		h = 0.0
	if h and h > 0.0:
		# h for luminance, hColor for color
		out = cv2.fastNlMeansDenoisingColored(out, None, h, h, 7, 21)

	# Auto white balance (gray-world approximation)
	if awb:
		# convert to float and scale channels so each channel mean equals overall mean
		f = out.astype(np.float32)
		b, g, r = cv2.split(f)
		mean_b = b.mean() if b.size else 1.0
		mean_g = g.mean() if g.size else 1.0
		mean_r = r.mean() if r.size else 1.0
		mean_all = (mean_b + mean_g + mean_r) / 3.0
		# avoid division by zero
		mean_b = mean_b if mean_b != 0 else 1.0
		mean_g = mean_g if mean_g != 0 else 1.0
		mean_r = mean_r if mean_r != 0 else 1.0
		b = b * (mean_all / mean_b)
		g = g * (mean_all / mean_g)
		r = r * (mean_all / mean_r)
		out = cv2.merge([b, g, r]).clip(0, 255).astype(np.uint8)

	# CLAHE on L channel to boost local contrast
	try:
		clip = float(clahe_clip)
	except Exception:
		clip = 2.0
	if clip and clip > 0.0:
		lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
		l, a, b = cv2.split(lab)
		clahe = cv2.createCLAHE(clipLimit=max(0.1, clip), tileGridSize=(8, 8))
		l2 = clahe.apply(l)
		lab2 = cv2.merge([l2, a, b])
		out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

	# Unsharp mask (sharpen)
	try:
		s = float(sharpen_amount)
	except Exception:
		s = 0.0
	if s and s > 0.0:
		blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=3)
		out = cv2.addWeighted(out, 1.0 + s, blurred, -s, 0)

	# Gamma correction
	try:
		gval = float(gamma)
	except Exception:
		gval = 1.0
	if gval and abs(gval - 1.0) > 1e-6:
		invGamma = 1.0 / gval if gval != 0 else 1.0
		table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
		out = cv2.LUT(out, table)

	return out


@app.route('/')
def index():
	return send_from_directory('.', 'index.html')


@app.route('/api/process', methods=['POST'])
def process_image():
	# Expects form-data: image=file, op=operation, and parameters depending on op
	if 'image' not in request.files:
		return jsonify({'error': 'no image file provided'}), 400
	file = request.files['image']
	img = read_image_file(file)
	if img is None:
		return jsonify({'error': 'invalid image'}), 400

	op = request.form.get('op', '')

	try:
		if op == 'brightness_contrast':
			alpha = float(request.form.get('alpha', 1.0))
			beta = float(request.form.get('beta', 0.0))
			out = adjust_brightness_contrast(img, alpha=alpha, beta=beta)
		elif op == 'grayscale':
			out = to_grayscale(img)
		elif op == 'blur':
			k = int(request.form.get('ksize', 5))
			out = blur_image(img, ksize=k)
		elif op == 'sharpen':
			out = sharpen_image(img)
		elif op == 'edge':
			method = request.form.get('method', 'canny')
			out = edge_detection(img, method=method)
		elif op == 'color_filter':
			typ = request.form.get('type', 'sepia')
			out = color_filter(img, typ)
		elif op == 'restore' or op == 'unfilter':
			# Attempt to reverse a prior color filter; requires the filter type param
			typ = request.form.get('type', 'sepia')
			out = undo_color_filter(img, typ)
		elif op == 'illustrate':
			# optional params: downscale, num_bilateral, edge_ksize (illustration/cartoon effect)
			downscale = int(request.form.get('downscale', 1))
			num_bilateral = int(request.form.get('num_bilateral', 7))
			edge_ksize = int(request.form.get('edge_ksize', 5))
			out = cartoonify_image(img, downscale=downscale, num_bilateral=num_bilateral, edge_ksize=edge_ksize)
		elif op == 'haze':
			# params: intensity (0..1 or 0..100), color (hex), vertical (0/1)
			try:
				int_raw = request.form.get('intensity', '0.5')
				intensity = float(int_raw)
			except Exception:
				intensity = 0.5
			# allow percent inputs (0..100)
			if intensity > 1.0:
				intensity = intensity / 100.0
			color = request.form.get('color', '#ffffff')
			vertical = str(request.form.get('vertical', '1')).lower() in ('1', 'true', 'yes')
			out = add_haze(img, intensity=intensity, color_hex=color, vertical=vertical)
		elif op == 'enhance':
			# parameters: fix_negative (0/1), denoise_h, clahe_clip, sharpen_amount, gamma, awb
			fix_neg = request.form.get('fix_negative', '0')
			fix_negative = str(fix_neg).lower() in ('1', 'true', 'yes')
			try:
				denoise_h = float(request.form.get('denoise_h', 0))
			except Exception:
				denoise_h = 0.0
			try:
				clahe_clip = float(request.form.get('clahe_clip', 2.0))
			except Exception:
				clahe_clip = 2.0
			try:
				sharpen_amount = float(request.form.get('sharpen_amount', 0.8))
			except Exception:
				sharpen_amount = 0.8
			try:
				gamma = float(request.form.get('gamma', 1.0))
			except Exception:
				gamma = 1.0
			awb = str(request.form.get('awb', '0')).lower() in ('1', 'true', 'yes')
			out = enhance_image(img, fix_negative=fix_negative, denoise_h=denoise_h, clahe_clip=clahe_clip, sharpen_amount=sharpen_amount, gamma=gamma, awb=awb)
		elif op == 'rotate':
			ang = float(request.form.get('angle', 0.0))
			out = rotate_image(img, ang)
		elif op == 'resize':
			w = int(request.form.get('w', img.shape[1]))
			h = int(request.form.get('h', img.shape[0]))
			out = resize_image(img, w, h)
		elif op == 'crop':
			x = int(request.form.get('x', 0))
			y = int(request.form.get('y', 0))
			w = int(request.form.get('w', img.shape[1]))
			h = int(request.form.get('h', img.shape[0]))
			out = crop_image(img, x, y, w, h)
		elif op == 'compress':
			# target: number; unit: kb or mb (case-insensitive). Optionally accept target as bytes when unit is 'b'
			target_raw = request.form.get('target', '')
			unit = request.form.get('unit', 'kb').lower()
			try:
				target_val = float(target_raw)
			except:
				return jsonify({'error': 'invalid target size'}), 400
			if unit in ('mb', 'm'):
				target_bytes = int(target_val * 1024 * 1024)
			elif unit in ('kb', 'k'):
				target_bytes = int(target_val * 1024)
			elif unit in ('b', 'bytes'):
				target_bytes = int(target_val)
			else:
				return jsonify({'error': 'unknown unit; use kb or mb'}), 400
			# call helper; honor fmt parameter
			fmt = request.form.get('fmt', 'jpg')
			data, out_fmt = compress_to_target(img, fmt=fmt, target_bytes=target_bytes)
			of = out_fmt.lower()
			if of in ('jpg', 'jpeg'):
				mimetype = 'image/jpeg'
			elif of == 'png':
				mimetype = 'image/png'
			elif of == 'webp':
				mimetype = 'image/webp'
			elif of in ('tif', 'tiff'):
				mimetype = 'image/tiff'
			else:
				mimetype = 'application/octet-stream'
			return send_file(BytesIO(data), mimetype=mimetype)
		else:
			out = img

		# Output format
		fmt = request.form.get('fmt', 'jpg')
		quality = int(request.form.get('quality', 90))
		data = cv2_to_bytes(out, fmt=fmt, quality=quality)
		# map fmt to standard mimetypes
		ff = fmt.lower()
		if ff in ('jpg', 'jpeg'):
			mimetype = 'image/jpeg'
		elif ff == 'png':
			mimetype = 'image/png'
		elif ff == 'webp':
			mimetype = 'image/webp'
		elif ff in ('tif', 'tiff'):
			mimetype = 'image/tiff'
		else:
			mimetype = 'application/octet-stream'
		return send_file(BytesIO(data), mimetype=mimetype)

	except Exception as e:
		return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
	# Use host=0.0.0.0 if you want external access; debug=False in production
	app.run(host='0.0.0.0', debug=True, port=5000)