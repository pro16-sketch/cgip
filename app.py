from io import BytesIO
import streamlit as st
import cv2
import numpy as np


def read_image_bytes(data):
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


def bgr_to_rgb_for_display(img):
	if len(img.shape) == 2:
		return img
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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


def run_operation(img, op, params):
	if op == 'brightness_contrast':
		return adjust_brightness_contrast(img, alpha=params['alpha'], beta=params['beta'])
	if op == 'grayscale':
		return to_grayscale(img)
	if op == 'blur':
		return blur_image(img, ksize=params['ksize'])
	if op == 'sharpen':
		return sharpen_image(img)
	if op == 'edge':
		return edge_detection(img, method=params['method'])
	if op == 'color_filter':
		return color_filter(img, params['type'])
	if op == 'restore':
		return undo_color_filter(img, params['type'])
	if op == 'illustrate':
		return cartoonify_image(
			img,
			downscale=params['downscale'],
			num_bilateral=params['num_bilateral'],
			edge_ksize=params['edge_ksize'],
		)
	if op == 'haze':
		return add_haze(
			img,
			intensity=params['intensity'],
			color_hex=params['color'],
			vertical=params['vertical'],
		)
	if op == 'enhance':
		return enhance_image(
			img,
			fix_negative=params['fix_negative'],
			denoise_h=params['denoise_h'],
			clahe_clip=params['clahe_clip'],
			sharpen_amount=params['sharpen_amount'],
			gamma=params['gamma'],
			awb=params['awb'],
		)
	if op == 'rotate':
		return rotate_image(img, params['angle'])
	if op == 'resize':
		return resize_image(img, params['w'], params['h'])
	if op == 'crop':
		return crop_image(img, params['x'], params['y'], params['w'], params['h'])
	return img


def main():
	st.set_page_config(page_title='Image Processing Studio', layout='wide')
	st.title('Image Processing Studio')
	st.caption('Streamlit app using OpenCV. No index.html dependency.')

	uploaded = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg', 'webp', 'tif', 'tiff'])
	if uploaded is None:
		st.info('Upload an image to begin.')
		return

	img = read_image_bytes(uploaded.getvalue())
	if img is None:
		st.error('Could not decode image.')
		return

	h, w = img.shape[:2]

	operations = [
		'none',
		'brightness_contrast',
		'grayscale',
		'blur',
		'sharpen',
		'edge',
		'color_filter',
		'restore',
		'illustrate',
		'haze',
		'enhance',
		'rotate',
		'resize',
		'crop',
		'compress',
	]

	st.sidebar.header('Controls')
	op = st.sidebar.selectbox('Operation', operations, index=0)
	params = {}

	if op == 'brightness_contrast':
		params['alpha'] = st.sidebar.slider('Contrast (alpha)', 0.0, 3.0, 1.0, 0.05)
		params['beta'] = st.sidebar.slider('Brightness (beta)', -100, 100, 0)
	elif op == 'blur':
		params['ksize'] = st.sidebar.slider('Kernel Size', 1, 31, 5, 2)
	elif op == 'edge':
		params['method'] = st.sidebar.selectbox('Method', ['canny', 'sobel'])
	elif op in ('color_filter', 'restore'):
		params['type'] = st.sidebar.selectbox('Filter Type', ['sepia', 'negative', 'warm', 'cool', 'vintage', 'emboss', 'sketch'])
	elif op == 'illustrate':
		params['downscale'] = st.sidebar.slider('Downscale', 1, 4, 1)
		params['num_bilateral'] = st.sidebar.slider('Bilateral Passes', 1, 12, 7)
		params['edge_ksize'] = st.sidebar.slider('Edge Kernel Size', 1, 15, 5, 2)
	elif op == 'haze':
		params['intensity'] = st.sidebar.slider('Haze Intensity', 0.0, 1.0, 0.5, 0.01)
		params['color'] = st.sidebar.color_picker('Haze Color', '#ffffff')
		params['vertical'] = st.sidebar.checkbox('Top-heavy haze', value=True)
	elif op == 'enhance':
		params['fix_negative'] = st.sidebar.checkbox('Fix Negative', value=False)
		params['denoise_h'] = st.sidebar.slider('Denoise', 0.0, 20.0, 0.0, 0.5)
		params['clahe_clip'] = st.sidebar.slider('CLAHE Clip', 0.0, 8.0, 2.0, 0.1)
		params['sharpen_amount'] = st.sidebar.slider('Sharpen', 0.0, 3.0, 0.8, 0.1)
		params['gamma'] = st.sidebar.slider('Gamma', 0.2, 3.0, 1.0, 0.05)
		params['awb'] = st.sidebar.checkbox('Auto White Balance', value=False)
	elif op == 'rotate':
		params['angle'] = st.sidebar.slider('Angle', -180.0, 180.0, 0.0, 1.0)
	elif op == 'resize':
		params['w'] = st.sidebar.number_input('Width', min_value=1, max_value=10000, value=int(w), step=1)
		params['h'] = st.sidebar.number_input('Height', min_value=1, max_value=10000, value=int(h), step=1)
	elif op == 'crop':
		params['x'] = st.sidebar.number_input('X', min_value=0, max_value=max(0, int(w - 1)), value=0, step=1)
		params['y'] = st.sidebar.number_input('Y', min_value=0, max_value=max(0, int(h - 1)), value=0, step=1)
		params['w'] = st.sidebar.number_input('Crop Width', min_value=1, max_value=int(w), value=int(w), step=1)
		params['h'] = st.sidebar.number_input('Crop Height', min_value=1, max_value=int(h), value=int(h), step=1)
	elif op == 'compress':
		params['target'] = st.sidebar.number_input('Target Size', min_value=1.0, value=300.0, step=10.0)
		params['unit'] = st.sidebar.selectbox('Unit', ['kb', 'mb', 'bytes'])
		params['fmt'] = st.sidebar.selectbox('Preferred Format', ['jpg', 'png', 'webp'])

	out_fmt = 'jpg'
	out_data = None
	processed = img

	try:
		if op == 'compress':
			target_val = float(params['target'])
			unit = params['unit']
			if unit == 'mb':
				target_bytes = int(target_val * 1024 * 1024)
			elif unit == 'kb':
				target_bytes = int(target_val * 1024)
			else:
				target_bytes = int(target_val)
			out_data, out_fmt = compress_to_target(img, fmt=params['fmt'], target_bytes=target_bytes)
			decoded = cv2.imdecode(np.frombuffer(out_data, np.uint8), cv2.IMREAD_UNCHANGED)
			if decoded is not None:
				if len(decoded.shape) == 2:
					processed = decoded
				elif decoded.shape[2] == 4:
					processed = cv2.cvtColor(decoded, cv2.COLOR_BGRA2BGR)
				else:
					processed = decoded
		else:
			if op != 'none':
				processed = run_operation(img, op, params)
			out_fmt = st.sidebar.selectbox('Output Format', ['jpg', 'png', 'webp', 'tiff'])
			quality = st.sidebar.slider('Quality', 1, 100, 90)
			out_data = cv2_to_bytes(processed, fmt=out_fmt, quality=quality)
	except Exception as err:
		st.error(f'Failed to process image: {err}')
		return

	left, right = st.columns(2)
	with left:
		st.subheader('Original')
		st.image(bgr_to_rgb_for_display(img), use_container_width=True)
	with right:
		st.subheader('Processed')
		st.image(bgr_to_rgb_for_display(processed), use_container_width=True)

	st.download_button(
		label='Download Processed Image',
		data=out_data,
		file_name=f'processed.{out_fmt}',
		mime='application/octet-stream',
	)


if __name__ == '__main__':
	main()