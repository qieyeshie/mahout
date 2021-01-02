package com.ldz;

import java.io.*;
import java.util.*;

public class ClassifyGender {

    static List<String> name_falemale = new ArrayList<>();
    static List<String> name_male = new ArrayList<>();
    static HashMap<String, Double> frequencyMap = new HashMap<>();
    static HashMap<String, Double> frequencyMap2 = new HashMap<>();
    static HashMap<String, Integer> name_map2 = new HashMap<>();
    static HashMap<String, Integer> name_map1 = new HashMap<>();
    static List<String> n_f_trainDataList = new ArrayList<>();
    static List<String> n_f_testDataList = new ArrayList<>();
    static List<String> n_m_trainDataList = new ArrayList<>();
    static List<String> n_m_testDataList = new ArrayList<>();

    //朴素贝叶斯算法
    static double base_f() {
        double mean_f = name_falemale.size() * 1.0 / (name_falemale.size() + name_male.size());
        double base_f = Math.log(mean_f);
        for (String k : frequencyMap.keySet()) {
            base_f += (Math.log(1 - frequencyMap.get(k)));
        }
        return base_f;
    }

    static double base_m() {
        double mean_m = name_male.size() * 1.0 / (name_falemale.size() + name_male.size());
        double base_m = Math.log(mean_m);
        for (String k : frequencyMap2.keySet()) {
            base_m += (Math.log(1 - frequencyMap2.get(k)));
        }
        return base_m;
    }

    static double GetLogProb_f(String string) {
        //拉普拉斯平滑参数
        double alpha = 1.0;
        //女性名字词频
        double freq_f = 0;
        //捕抓空值异常
        try {
            freq_f = name_map1.get(string);
        } catch (NullPointerException nullPointerException) {

        } finally {

        }
        //拉普拉斯平滑计算
        double freq_smooth_f = (freq_f * 1.0 + alpha) / (name_map1.size() + frequencyMap.size() * alpha);
        //返回平滑处理后的结果
        return Math.log(freq_smooth_f) - Math.log(1 - freq_smooth_f);
    }

    static double GetLogProb_m(String string) {
        double alpha = 1.0;
        double freq_m = 0;
        //捕抓空值异常
        try {
            freq_m = name_map2.get(string);
        } catch (NullPointerException nullPointerException) {

        } finally {

        }
        double freq_smooth_m = (freq_m * 1.0 + alpha) / (name_map2.size() + frequencyMap2.size() * alpha);
        return Math.log(freq_smooth_m) - Math.log(1 - freq_smooth_m);
    }

    //朴素贝叶斯算法
    static boolean ComputerLogProb(String string) {
        double logprob_f = base_f();
        double logprob_m = base_m();
        String[] splitt = string.split("");
        for (String k : splitt) {
            logprob_f += GetLogProb_f(k);
            logprob_m += GetLogProb_m(k);
        }
        //返回结果（true表示为女名，false为男名）
        return (logprob_f > logprob_m);
    }


    static int Getgender(String string) {
        int g = 0;
        if (ComputerLogProb(string) == true) {
            g = 0;
        } else {
            g = 1;
        }
        return (g);
    }


    public static void main(String[] args) throws IOException {

        //读取数据
        DataInputStream file = new DataInputStream(new FileInputStream(new File("E:\\mahout\\day3\\data\\name_data.txt")));
        BufferedReader reader = new BufferedReader(new InputStreamReader(file, "UTF-8"));
        HashMap<String, String> tokenMap = new HashMap<>();
        List<String> testDataList = new ArrayList<>();
        List<String> names_falemale = new ArrayList<>();
        List<String> names_male = new ArrayList<>();
        String line = reader.readLine();
        while ((line = reader.readLine()) != null) {
            String[] lineArr = line.split(",");
            tokenMap.put(lineArr[1], lineArr[2]);
            int gender = Integer.parseInt(lineArr[2]);
            if (gender == 0) {
                name_falemale.add(lineArr[1]);
            } else {
                name_male.add(lineArr[1]);
            }
        }

        //七成数据做训练集
        float b = 0.7f;
        n_f_trainDataList = name_falemale.subList(0, (int) (b * name_falemale.size()));
        n_m_trainDataList = name_male.subList(0, (int) (b * name_male.size()));
        n_f_testDataList = name_falemale.subList((int) (b * name_falemale.size()), name_falemale.size());
        n_m_testDataList = name_male.subList((int) (b * name_male.size()), name_male.size());
        for (String k : n_f_testDataList) {
            testDataList.add(k);
        }
        for (String s : n_m_testDataList) {
            testDataList.add(s);
        }

        //女性名字分词得到names_falemale
        for (String name1 : n_f_trainDataList) {
            String[] surname1 = name1.split("");
            for (String c1 : surname1) {
                names_falemale.add(c1);
            }
        }


        //男性分词的得到names_male
        for (String name2 : n_m_trainDataList) {
            String[] surname2 = name2.split("");
            for (String c2 : surname2) {
                names_male.add(c2);
            }
        }


        //统计女性名字分词后的每个文字的数量
        for (int i = 0; i < names_falemale.size(); i++) {
            String ch1 = names_falemale.get(i);
            if (!name_map1.containsKey(ch1)) {
                name_map1.put(ch1, 1);
            } else {
                int val1 = name_map1.get(ch1);
                val1++;
                name_map1.put(ch1, val1);
            }
        }


        //统计男性名字分词后的每个文字的数量
        for (int i = 0; i < names_male.size(); i++) {
            String ch2 = names_male.get(i);
            if (!name_map2.containsKey(ch2)) {
                name_map2.put(ch2, 1);
            } else {
                int val2 = name_map2.get(ch2);
                val2++;
                name_map2.put(ch2, val2);
            }
        }

        //测试结果
        int correctNum = 0;
        for (String k : testDataList) {
            if (Integer.parseInt(tokenMap.get(k)) == (Getgender(k))) {
                ++correctNum;
            }
        }
        //正确率
        System.out.printf("correctNum = %d correct precent = %f \n", correctNum, correctNum * 1.0 / (testDataList.size()));
    }

}








