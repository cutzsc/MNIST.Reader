using System;
using System.Drawing;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing.Imaging;

namespace MNIST.Reader
{
	public class MNISTData
	{
		public readonly int count;
		public readonly int rows;
		public readonly int cols;

		byte[] pixels;
		byte[] labels;
		List<int>[] indexBuffer;
		int imageSize;

		Random rand;
		int caret = 0;
		int pos = -1;

		public MNISTData(int imagesCount, int imageWidth, int imageHeight)
		{
			rand = new Random();
			count = imagesCount;
			cols = imageWidth;
			rows = imageHeight;
			imageSize = imageWidth * imageHeight;
			labels = new byte[count];
			pixels = new byte[count * imageWidth * imageHeight];
			indexBuffer = new List<int>[10];
			for (int i = 0; i < 10; i++)
				indexBuffer[i] = new List<int>();
		}

		public void SetData(byte[] pixels, byte[] labels)
		{
			if (pixels.Length != this.pixels.Length ||
				labels.Length != this.labels.Length)
				throw new ArgumentException();

			for (int i = 0, offset = 0; i < count; i++, offset += imageSize)
			{
				Array.Copy(pixels, offset, this.pixels, offset, imageSize);
				this.labels[i] = labels[i];
				indexBuffer[labels[i]].Add(offset);
			}

			caret = count;
		}

		public void Add(byte[] image, byte label)
		{
			if (image.Length != imageSize)
				throw new ArgumentException();

			if (caret == count)
				throw new OverflowException();

			int offset = caret * imageSize;
			Array.Copy(image, 0, pixels, offset, imageSize);
			indexBuffer[label].Add(offset);
			caret++;
		}

		public byte[] ImageAsPixels(byte label)
		{
			byte[] img = new byte[imageSize];
			Array.Copy(pixels, RandomIndex(label), img, 0, imageSize);
			return img;
		}

		public Bitmap Image(byte label, PixelFormat format = PixelFormat.Format32bppArgb)
		{
			Bitmap bmp = new Bitmap(cols, rows, format);
			int offset = indexBuffer[label][RandomIndex(label)];
			for (int y = 0, px = offset; y < rows; y++)
			{
				for (int x = 0; x < cols; x++, px++)
				{
					bmp.SetPixel(x, y,
						Color.FromArgb(pixels[px], pixels[px], pixels[px], pixels[px]));
				}
			}
			return bmp;
		}

		private int RandomIndex(byte label)
		{
			return rand.Next(0, indexBuffer[label].Count);
		}
	}
}
