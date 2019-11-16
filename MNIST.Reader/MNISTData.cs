using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Threading.Tasks;

namespace KernelDeeps.IO.MNIST
{
	public class MNISTData
	{
		public readonly int count;
		public readonly int rows;
		public readonly int cols;

		int imageSize;
		byte[] pixels;
		byte[] labels;
		List<int>[] indexBuffer;
		Bitmap[] bitmapBuffer;

		Random rand;

		public MNISTData(int imagesCount, int imageWidth, int imageHeight, byte[] pixels, byte[] labels)
		{
			if (pixels.Length != imagesCount * imageWidth * imageHeight ||
				labels.Length != imagesCount)
				throw new ArgumentException();

			rand = new Random();
			count = imagesCount;
			cols = imageWidth;
			rows = imageHeight;
			imageSize = imageWidth * imageHeight;
			this.labels = new byte[count];
			this.pixels = new byte[count * imageWidth * imageHeight];
			indexBuffer = new List<int>[10];
			for (int i = 0; i < 10; i++)
				indexBuffer[i] = new List<int>();

			for (int i = 0, offset = 0; i < count; i++, offset += imageSize)
			{
				Array.Copy(pixels, offset, this.pixels, offset, imageSize);
				this.labels[i] = labels[i];
				indexBuffer[labels[i]].Add(offset);
			}

			Array.ForEach(indexBuffer, (item) => item.Capacity = item.Count);
		}

		public void CreateBitmapBuffer()
		{
			bitmapBuffer = new Bitmap[count];

			Parallel.For(0, count, new ParallelOptions() { MaxDegreeOfParallelism = Environment.ProcessorCount }, (i) =>
			{
				int offset = i * imageSize;
				bitmapBuffer[i] = new Bitmap(cols, rows, PixelFormat.Format32bppArgb);
				for (int y = 0; y < rows; y++)
				{
					for (int x = 0; x < cols; x++)
					{
						int px = 255 - pixels[offset++];
						bitmapBuffer[i].SetPixel(x, y, Color.FromArgb(px, px, px));
					}
				}
			});
		}

		public void DeleteBitmapBuffer()
		{
			foreach (Bitmap bmp in bitmapBuffer)
				bmp.Dispose();
			bitmapBuffer = null;
		}

		public int GetGlobalIndex(Label label, int index)
		{
			return indexBuffer[(int)label][index] / imageSize;
		}

		public byte GetLabel(int index)
		{
			return labels[index];
		}

		#region normalized data

		public float[][][] NormalizedDataFloat()
		{
			float[][][] data = new float[count][][];
			Parallel.For(0, count, new ParallelOptions() { MaxDegreeOfParallelism = Environment.ProcessorCount }, (i) =>
			{
				data[i] = new float[2][];
				data[i][0] = new float[imageSize];

				int offset = i * imageSize;
				for (int px = 0; px < imageSize; px++)
				{
					data[i][0][px] = pixels[offset + px] / 255.0f;
				}

				data[i][1] = new float[10];
				data[i][1][labels[i]] = 1f;
			});
			return data;
		}

		public double[][][] NormalizedDataDouble()
		{
			double[][][] data = new double[count][][];
			Parallel.For(0, count, new ParallelOptions() { MaxDegreeOfParallelism = Environment.ProcessorCount }, (i) =>
			{
				data[i] = new double[2][];
				data[i][0] = new double[imageSize];

				int offset = i * imageSize;
				for (int px = 0; px < imageSize; px++)
				{
					data[i][0][px] = pixels[offset + px] / 255.0d;
				}

				data[i][1] = new double[10];
				data[i][1][labels[i]] = 1d;
			});
			return data;
		}

		public float[] NormalizedPixelsFloat()
		{
			float[] pixels = new float[this.pixels.Length];
			Array.Copy(this.pixels, pixels, pixels.Length);
			Array.ForEach(pixels, px => px /= 255f);
			return pixels;
		}

		public double[] NormalizedPixelsDouble()
		{
			double[] pixels = new double[this.pixels.Length];
			Array.Copy(this.pixels, pixels, pixels.Length);
			Array.ForEach(pixels, px => px /= 255d);
			return pixels;
		}

		public float [] NormalizedLabelsFloat()
		{
			float[] labels = new float[this.labels.Length];
			Array.Copy(this.labels, labels, labels.Length);
			Array.ForEach(labels, label => label /= 255f);
			return labels;
		}

		public double[] NormalizedLabelsDouble()
		{
			double[] labels = new double[this.labels.Length];
			Array.Copy(this.labels, labels, labels.Length);
			Array.ForEach(labels, label => label /= 255f);
			return labels;
		}

		#endregion

		#region byte[] image

		/// <summary>
		/// Get random image data.
		/// </summary>
		public byte[] ImageData(out int index)
		{
			index = RandomGlobalIndex();
			return ImageData(index);
		}

		/// <summary>
		/// Get random image that belongs to label "num".
		/// </summary>
		public byte[] ImageData(Label num)
		{
			return ImageData(num, RandomLabeledIndex(num));
		}

		/// <summary>
		/// Get image data with label "num" on "index" position.
		/// </summary>
		public byte[] ImageData(Label num, int index)
		{
			byte[] data = new byte[imageSize];
			Array.Copy(pixels, indexBuffer[(int)num][index], data, 0, imageSize);
			return data;
		}

		/// <summary>
		/// Get image data thats belong to [1, ..., index, ..., n] of images.
		/// </summary>
		public byte[] ImageData(int index)
		{
			byte[] data = new byte[imageSize];
			Array.Copy(pixels, index * imageSize, data, 0, imageSize);
			return data;
		}

		#endregion

		#region Bitmap image

		/// <summary>
		/// Get random image.
		/// </summary>
		public Bitmap Image(out int index)
		{
			index = RandomGlobalIndex();
			return Image(index);
		}

		/// <summary>
		/// Get random image that belongs to label "num".
		/// </summary>
		public Bitmap Image(Label num)
		{
			return Image(num, RandomLabeledIndex(num));
		}

		/// <summary>
		/// Get image with label "num" on "index" position.
		/// </summary>
		public Bitmap Image(Label num, int index)
		{
			if (index < 0 ||
				index > indexBuffer[(int)num].Count)
				throw new ArgumentException();

			if (bitmapBuffer != null)
				return (Bitmap)bitmapBuffer[GetGlobalIndex(num, index)].Clone();

			Bitmap bmp = new Bitmap(cols, rows, PixelFormat.Format32bppArgb);
			int offset = indexBuffer[(int)num][index];
			for (int y = 0; y < rows; y++)
			{
				for (int x = 0; x < cols; x++)
				{
					byte px = (byte)(byte.MaxValue - pixels[offset++]);
					bmp.SetPixel(x, y, Color.FromArgb(px, px, px));
				}
			}
			return bmp;
		}

		/// <summary>
		/// Get image thats belong to [1, ..., index, ..., n] of images.
		/// </summary>
		public Bitmap Image(int index)
		{
			if (index < 0 ||
				index > count)
				throw new ArgumentException();

			if (bitmapBuffer != null)
				return (Bitmap)bitmapBuffer[index].Clone();

			Bitmap bmp = new Bitmap(cols, rows, PixelFormat.Format32bppArgb);
			int offset = index * imageSize;
			for (int y = 0; y < rows; y++)
			{
				for (int x = 0; x < cols; x++)
				{
					byte px = (byte)(byte.MaxValue - pixels[offset++]);
					bmp.SetPixel(x, y, Color.FromArgb(px, px, px));
				}
			}
			return bmp;
		}

		#endregion

		private int RandomGlobalIndex()
		{
			return rand.Next(0, count);
		}

		private int RandomLabeledIndex(Label num)
		{
			return rand.Next(0, indexBuffer[(int)num].Count);
		}

		#region Enumerator

		int position = -1;

		public byte CurrentLabel => labels[position];
		public byte[] CurrentImageData => ImageData(position);
		public Bitmap CurrentImage => Image(position);

		public void Reset()
		{
			position = -1;
		}

		public bool MoveNext()
		{
			if (position < count - 1)
			{
				position++;
				return true;
			}
			else
				return false;
		}

		public double[][] CurrentNormalizedSetDouble()
		{
			double[][] data = new double[2][];
			int offset = position * imageSize;
			data[0] = new double[imageSize];
			for (int i = 0; i < imageSize; i++)
			{
				data[0][i] = pixels[offset + i] / 255d;
			}
			data[1] = new double[10];
			data[1][labels[position]] = 1d;
			return data;
		}

		public float[][] CurrentNormalizedSetFloat()
		{
			float[][] data = new float[2][];
			int offset = position * imageSize;
			data[0] = new float[imageSize];
			for (int i = 0; i < imageSize; i++)
			{
				data[0][i] = pixels[offset + i] / 255f;
			}
			data[1] = new float[10];
			data[1][labels[position]] = 1f;
			return data;
		}

		#endregion
	}
}
