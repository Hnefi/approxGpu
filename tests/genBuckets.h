#ifndef _GENBUCKETS_H
#define _GENBUCKETS_H

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <assert.h>

// boost for scalability!
#include <boost/unordered_map.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/foreach.hpp>
#include <boost/container/deque.hpp>
#include <boost/container/vector.hpp>
#include <boost/container/set.hpp>
#include <boost/algorithm/string_regex.hpp>
#include <boost/algorithm/string/erase.hpp>
#include <boost/range/algorithm/sort.hpp>
#include <boost/random.hpp>
#include <boost/math/special_functions/round.hpp>

// ultra optimized hash function for big integer tables
struct IntegerHasher 
{
    size_t operator()(const int& k) const
    {
        size_t ret = ((k >> 16) ^ k) * 0x45d9f3b;
        ret = ((ret >> 16) ^ ret) * 0x45d9f3b;
        ret = ((ret >> 16) ^ ret);
        return ret;
    }
};

//TODO: Compare floats?!?!?! I'm not sure you're supposed to do this.
typedef boost::container::deque<float> ThreadDq;
typedef boost::container::vector<float> FloatVector;
typedef boost::container::set<float> FloatSet;
typedef boost::shared_ptr<ThreadDq> DqPtr;
typedef boost::unordered_map<int,DqPtr,IntegerHasher> ThreadToValueMap;
typedef boost::unordered_map<float,DqPtr> GHBHashToValueMap;
typedef boost::random::mt19937 _BASE_GENERATOR;
typedef boost::random::uniform_int_distribution<> distribution_type;

class ValueLocStats
{
    //TODO
};

template <typename MapType>
class ValueReader
{
    private:
        MapType& mapToEdit;
        std::ifstream& myInput;

        void dumpToFile(void) {
            std::ofstream dumpFile;
            dumpFile.open("GHBHashValMap.txt");
            typename MapType::iterator it = mapToEdit.begin();
            for(; it != mapToEdit.end(); it++) {
                std::pair<float,DqPtr> v = *it;
                dumpFile << "Key: " << v.first << std::endl;
                DqPtr this_dq = v.second;
                for(int i = 0; i < this_dq->size() ; i++)
                    dumpFile << "\tval[" << i << "] = " << this_dq->at(i) << std::endl;
            }
            dumpFile.close();
        }

        void processLine(std::string &in) {
            if(in.empty()) return;
            // use boost algos to find the thread idx: value between two [ ] exprs
            boost::regex thrNumReg("(\\[\\d+\\.\\d+\\])",boost::regex::egrep);
            boost::iterator_range<std::string::iterator> ret = find_regex(in,thrNumReg);
            //std::cout << "reg result: " << ret << std::endl;
            std::string filtered(ret.begin(),ret.end());
            boost::algorithm::erase_all(filtered,"[");
            boost::algorithm::erase_all(filtered,"]");
            // convert to int
            float hash = atof(filtered.c_str());
            hash /= (float)225.0;
            //assert(hash < 1.0);

            // use boost algos to find what's after the colon (float)
            boost::algorithm::erase_all(in," ");
            boost::regex fp("(:\\d+\\.\\d+)",boost::regex::egrep);
            ret = find_regex(in,fp);
            //std::cout << "reg result: " << ret << std::endl;
            filtered = std::string(ret.begin(),ret.end());
            boost::algorithm::erase_all(filtered,":");
            float val = atof(filtered.c_str());
            //std::cout << "val: " << val << std::endl;

            // now that we have the values, we can play around with the big hash table.
            // (This way avoids a call to find and then insert.)
            std::pair<typename MapType::iterator,bool> insert_it;
            DqPtr new_dq( new ThreadDq() );
            insert_it = mapToEdit.insert( std::make_pair(hash,new_dq) );
            if(insert_it.second == false) {
                DqPtr old_dq = (insert_it.first)->second; // haha this is a funny looking call
                old_dq->push_back(val);
            } else {
                new_dq->push_back(val);
            }
        }

    public:
        ValueReader(MapType& usethis, std::ifstream& in) : mapToEdit(usethis), myInput(in) { }

        // master function that reads this istream and makes a map of all the values that come in
        void processInput(void) {
            if ( myInput.is_open() ) {
                std::string raw_string;
                while( !myInput.eof() ) {
                    getline(myInput,raw_string);
                    processLine(raw_string);
                }
            }
        }

        void dumpMap(bool toFile = false) {
            if(toFile) dumpToFile();
            else {
                typename MapType::iterator it = mapToEdit.begin();

                //for(auto& v : mapToEdit) { // over each deque
                for(; it != mapToEdit.end(); it++) {
                    std::pair<float,DqPtr> v = *it;
                    std::cout << "Key: " << v.first << std::endl;
                    DqPtr this_dq = v.second;
                    for(int i = 0; i < this_dq->size() ; i++)
                        std::cout << "\tval[" << i << "] = " << this_dq->at(i) << std::endl;
                }
            }
        }

        void reduceAverages(void) {
            typename MapType::iterator it = mapToEdit.begin();

            for(; it != mapToEdit.end(); it++) {
                std::pair<float,DqPtr> v = *it;
                DqPtr this_dq = v.second;
                if(this_dq->size() == 1) continue; // no need to reduce
                else {
                    std::cout << "In reduce ave, key: " << v.first << " has " << this_dq->size() << " elements." << std::endl;
                    float run_sum = 0.0;
                    float num_elem = (float) this_dq->size();
                    for(int i = this_dq->size()-1; i >= 0; i--) {
                        float val = this_dq->at(i);
                        if (val < 300) {
                            run_sum += this_dq->at(i);
                        }
                        this_dq->pop_back();
                    }
                    float write = run_sum / num_elem;
                    this_dq->push_back(write);
                    std::cout << "\trun_sum = " << run_sum << std::endl;
                    std::cout << "\tWriting to make key/v pair: [" << v.first << "," << this_dq->at(0) << "]" << std::endl;
                    //assert(this_dq->at(0) < 1.0);
                }
            }
        }

};

class LVAData
{
    private:
        GHBHashToValueMap myMap;
        ValueReader<GHBHashToValueMap> slaveReader;
        std::ifstream inFile;

    public:
        LVAData(std::string input_file) : inFile(input_file.c_str()), slaveReader(myMap,inFile) { }
        FloatVector GenerateUniqueValueSet(unsigned int max_num);

        void PrintTables() {
            slaveReader.dumpMap(true);
        }
};

#endif // #ifndef _GENBUCKETS_H
