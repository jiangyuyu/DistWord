package discword;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class FileUtil {
	public static String read(String fpath){
		try {
			BufferedReader reader = new BufferedReader(new FileReader(fpath));
			String line;
			String text = "";
			try {
				while((line = reader.readLine()) != null){
					text += line;
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
			return text;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return null;
	}
}
