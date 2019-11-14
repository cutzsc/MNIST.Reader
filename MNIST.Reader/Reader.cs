using System;
using System.IO;

namespace KernelDeeps.IO.MNIST
{
	public static class MNISTReader
    {
		public static MNISTData Read(string labelsPath, string imagesPath)
		{
			BinaryReader reader = new BinaryReader(File.OpenRead(labelsPath));

			int l_magic = NextInt32(reader);
			int l_size = NextInt32(reader);

			byte[] labels = new byte[l_size];
			reader.Read(labels, 0, l_size);

			reader.Close();
			reader.Dispose();

			reader = new BinaryReader(File.OpenRead(imagesPath));

			int magic = NextInt32(reader);
			int size = NextInt32(reader);
			int rows = NextInt32(reader);
			int cols = NextInt32(reader);

			byte[] pixels = new byte[size * rows * cols];
			reader.Read(pixels, 0, pixels.Length);

			reader.Close();
			reader.Dispose();

			return new MNISTData(size, cols, rows, pixels, labels);
		}

		private static int NextInt32(BinaryReader reader)
		{
			byte[] buffer = reader.ReadBytes(4);
			if (BitConverter.IsLittleEndian)
				Array.Reverse(buffer);
			return BitConverter.ToInt32(buffer, 0);
		}
    }
}
