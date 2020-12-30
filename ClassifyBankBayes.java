package com.digquant;

import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.StandardNaiveBayesClassifier;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ClassifyBankBayes {
    public static HashMap<String, Integer> jobMap = new HashMap<>();

    public static void main(String[] args) throws IOException {
        // 1, 加载数据
        String path = "F:\\mahout\\day3\\data\\bank-full.csv";
        File file = new File(path);
        BufferedReader reader = new BufferedReader(new FileReader(file));
        // 1.1 去掉第一行标题
        String line = reader.readLine();
        int jobIndex = 0;
        while ((line = reader.readLine()) != null) {
            String[] lineArr = line.split(";");
            jobMap.put(lineArr[1], jobIndex++);
            System.out.println(jobMap);
        }
    }

}