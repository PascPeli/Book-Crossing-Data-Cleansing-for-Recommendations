# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:26:36 2018

@author: PascPeli
"""

import pandas as pd
import os
import sys
import logging
import surprise
#from BookRec_Functions import *


class recommender:
    
    def __init__(self):
        
        self.root_dir = os.getcwd()
        self.dfs_path = os.path.join(self.root_dir, 'Data/datasets/new datasets/')
        self.model_path = os.path.join(self.root_dir, 'Data/model.pickle')
        self.users_df, self.items_df, self.ratings_df = self.load_dfs()
        self.pos, self.neg = ['y','yes','Y','Yes'], ['n','no','N','No']
        self.nof_user_ratings = self.ratings_df.user_id.value_counts()
        self.min_nof_ratings = 1
        self.ratings_changed = False
        
        # Check if there is a save of the model and if yes, load it. If not, train it now
        try:
            _, self.algorithm = surprise.dump.load(self.model_path)
        except:
            logging.error(('File "model.pickle" was not found in %s.\n If you have already '
                           'trained the Recommender, make sure the file is in the correct directory'), self.model_path)
            train_flag = self.input_y_n('Would you like to train the Recommender again (y/n)? ')
            if train_flag in self.pos:
                self.model_fit()

                
    
    def input_y_n(self, message, message_loop='', additional=[]):
        '''
        Ask for input and check if it is valid (is in self.pos, self.neg and additional if provided)
        
        Args:
            message (str): String to be used as [prompt] in the initial input
            message_loop (str): String to be used as [prompt] in the looped input 
            additional (list): List of additional valid input strings
        
        Returns:
            inp (str): keyboard provided string input. One of self.pos, self.neg and additional if provided.
        '''
        if message_loop=='':
            message_loop = message
        inp = input (message)
        while inp not in self.pos + self.neg + additional:
            print('Incorrect input.')
            inp = input(message_loop)
        return inp    

    
    def main_menu(self):
        '''
        Main menu function to choose system actions based on user input.
        '''
        print('Welcome! This is a Recommender System build to recommend Books. It uses the Book-Crossing Dataset'
              'mined by Cai-Nicolas Ziegler. The dataset has been processed using "BookCrossing data cleansing.ipynb".')
        #print('Since the dataset was sparse it works best if you rate a lot of items.')
        keep_on_mm = True
        while keep_on_mm:
            print('What would you like to do?\n Choose a number to...\n','1:Rate Books   2:Get Recommendations   3:Logout   4:Quit')
            mm_input = input(':')
            while mm_input not in ['1','2','3','4','#']:
                mm_input = input('Choose a number between 1 and 4 to Rate Books, Get Recommendation, Logout or Quit')
            if mm_input == '1':
                self.new_ratings()
            elif mm_input == '2':
                self.recommend()
            elif mm_input == '3':
                self.user_logout()
                print('You have logged out Successfully')
            elif mm_input == '4':
                self.save_dfs('all')
                self.save_model(verbose=False)
                keep_on_mm = False
                print('We hope you enjoyed the experience and to see you again soon...Bye...')
            else:
                print('You have entered Advanced Settings')
                a_ch = input('')
                if a_ch in ['SVD','Baseline','SlopeOne','KNNBasic']:
                    self._algo_choise = a_ch
            
    
    def user_login(self):
        '''
        Check if user already registered and log him in. If not registered create a new user if he wants.
        '''
        try:
            self.user_Id
            right_id = self.input_y_n('Is your user ID : {0} (y/n)? '.format(self.user_Id))
            if right_id in self.neg:
                self.user_logout()
                self.user_login()
        except:        
            al_u = self.input_y_n('Are you a user already?? (y/n) : ', 'Please type "y" for yes or "n" for no.\n Are you a user already?? (y/n) : ')
    
            if al_u in self.pos:
                u_Id_in = input('Insert your user ID : ')
                while (not u_Id_in.isdigit()) or (int(u_Id_in) not in self.users_df.user_id.values):
                    if (not u_Id_in.isdigit()):
                        u_Id_in = input('HINT: It is a number!!! Insert your user ID : ')
                    else:
                        u_Id_in = input('We don\'t seem to be able to find you in the database.\n' 
                                        'If you are a user please insert you valid user ID.\n'
                                        'Or if you are not a user already please press "r" to register : ')
                        if u_Id_in =='r':
                            print('It seems we have a new User')
                            self.create_new_user()
                            break  
                try:
                    self.user_Id
                except:
                    self.user_Id = int(u_Id_in)
            elif al_u in self.neg:
                print('It seems we have a new User')
                self.create_new_user()
            
            print('Welcome user ', self.user_Id)
 

    def user_logout(self):
        '''
        Logout user
        '''
        try:
            del self.user_Id
        except:
            print('You are not logged in')
                
                
    def create_new_user(self):
        '''
        Create a new user and assign him with a user Id.
        '''
    
        # get new users age and check if it is valid 
        age = input ('Please insert your age : ')
        while (not age.isdigit()) or (int(age)<5 or int(age)>100):
            age = input ('Please insert your age (it should  be a number between 5 and 100) : ')
        age = int(age)
        
        # get new users location as a comma separated str, split it and strip it of unnecessary spaces 
        location = input ('Please insert your City, State/Province and Country separated by "," (e.g "Birmingham, West Midlands, United Kingdom") :\n')
        csc=['','','']
        for i, value in enumerate(location.split(',')):
            if i<3:
                csc[i] = value.strip()
        
        # give new user the next available ID
        self.user_Id = max(int(self.users_df.user_id))+1
        
        # update users_df
        new_user = pd.DataFrame([self.user_Id, csc[0],csc[1],csc[2], age]).T
        new_user.columns= self.users_df.columns
        self.users_df = pd.concat([self.users_df, new_user], axis=0, ignore_index=True)
        #self.save_dfs('users')
        
        print('\n','_-_-_-_-!*!-_-_-_-'*5)
        print('\nYou are now registered!!! Your user ID is %d. '
              'Please remember it since currently there is no way to retrieve it.' %self.user_Id)
        self.nof_user_ratings = self.ratings_df.user_id.value_counts()
    

    def search_items(self):
        '''
        Search items based on an input search string 
        
        Return: 
            items: pandas.DataFrame of items corresponding to the search string (input). 
        '''    
        # Ask for key and string to search for
        by = input ('Search items based on %s : ' %self.items_df.columns.values)
        while by not in self.items_df.columns:
            by = input ('Incorrect input. Type one of the following to search items by, %s : ' %self.items_df.columns.values)
        search_str = input ('Search for items with %s equal to : ' %by)
        
        if self.items_df[by].dtype != object:
            search_str = int(search_str)
        
        # get items corresponding to search string
        items = self.items_df.loc[self.items_df[by]==search_str]
        if items.empty:
            print ('There are no items with %s equal to "%s"' %(by, str(search_str)))
        return items.reset_index(drop=True)


    def new_ratings (self):
        '''
        Use search_item method to get items and rate them. Can rate one or multiple items.
        '''
        # check if user is logged in and then search and rate items until instructed otherwise
        self.user_login()
        keep_on_r = True
        while keep_on_r:
            items = self.search_items()
            if not items.empty:
                print('The items found in our database based on your search is ', items)
                rate_flag = self.input_y_n('Would you like to rate any of them? (y/n) ')
                if rate_flag in self.pos:
                    if len(items) == 1:
                        index = ['0']
                    else:    
                        err_flag = True
                        # check if input is valid and indexes exist
                        while err_flag:
                            index = input('Please insert the index/es for the item/s you would like to rate. \n'
                                      '(If more than one separate the indexes with commas ",") : ').split(',')
                            count=0
                            for idx in index:
                                if (idx.isdigit()) and (int(idx)>0 and int(idx)<len(items)):
                                    count+=1
                            if count==len(index): 
                                err_flag=False
                            else:
                                print('Invalid input. Please insert number between 0 and ', len(items)-1)
                    
                    # construct lists and then dfs of new ratings
                    index = [int(x.strip()) for x in index]
                    user, isbn, rating = [], [], []
                    for idx in index:
                        rat = input ('What\'s your rating for the movie with index {0} : '.format(idx))
                        while (not rat.isdigit()) or (int(rat)<=0 or int(rat)>10):
                            rat = input ('That was not a valid rating. Ratings should be a number between 1 and 10. \n'
                                         'What\'s your rating for the movie with index {0} : '.format(idx))
                        rating.append(rat)
                        user.append(self.user_Id)
                        isbn.append(items.loc[idx,'isbn'])                   
                    new_ratings = pd.DataFrame([user, isbn, rating]).T
                    new_ratings.columns = self.ratings_df.columns
                    self.ratings_df = pd.concat([self.ratings_df, new_ratings],axis=0, ignore_index=True)
                    self.ratings_changed=True
            
            # Ask if user wants to keep on searching and rating                
            new_search = self.input_y_n('Would you like to perform a new search? (y/n) : ')            
            if new_search in self.neg:
                keep_on_r = False
            self.nof_user_ratings = self.ratings_df.user_id.value_counts()
            


    def model_fit (self):
        '''
        Train model using surprise.SVD algorithm. 
        '''
        self.build_trainset()
        algo = self._algo_choise
        if algo == 'SVD':
            self.algorithm = surprise.SVD()
        elif algo == 'Baseline':
            self.algorithm = surprise.BaselineOnly()
        elif algo == 'SlopeOne':
            self.algorithm = surprise.SlopeOne()
        else:
            self.algorithm = surprise.KNNBasic()
        
        print('Training Recommender System using %s...' %algo)
        
        self.algorithm.fit(self.trainset)
        self.ratings_changed=False
        print('Done')
        
                
    def save_model (self, verbose=True):
        '''
        Save model in ../Data.
        
        Args:
            verbose (bool): Level of verbosity. If 1, then a message indicates that the dumping went successfully. Default is 0
        '''
        if verbose:
            print('Saving Model...')
        verbose=1*verbose
        surprise.dump.dump(self.model_path, predictions=None, algo=self.algorithm, verbose=verbose)
        
        
    def build_trainset(self):
        '''
        Build the trainset from ratings_df to be used by the <surprise.prediction_algorithms.algo_base.AlgoBase>.fit()
        '''
        reader = surprise.Reader(rating_scale=(1, 10))
        data = surprise.Dataset.load_from_df(self.ratings_df[['user_id', 'isbn', 'rating']], reader)
        self.trainset = data.build_full_trainset()
    
    
    def build_recset(self, trainset, fill=None):
        '''
        Return a list of ratings that can be used as a testset in the
        :meth:`test() <surprise.prediction_algorithms.algo_base.AlgoBase.test>`
        method. The ratings are all the ratings that are **not** in the trainset, i.e.
        all the ratings :math:`r_{ui}` where the user :math:`u` is known, the
        item :math:`i` is known, but the rating :math:`r_{ui}`  is not in the
        trainset. As :math:`r_{ui}` is unknown, it is either replaced by the
        :code:`fill` value or assumed to be equal to the mean of all ratings
        :meth:`global_mean <surprise.Trainset.global_mean>`.

        Args:
            trainset (surprise.Trainset.obj) -- The trainset used to fit/train the model.
            fill(float) -- The value to fill unknown ratings. If :code:`None` the
                global mean of all ratings :meth:`global_mean
                <surprise.Trainset.global_mean>` will be used.

        Returns:
            A list of tuples ``(uid, iid, fill)`` where ids are raw ids.
        '''
        trainset = self.trainset
        fill = trainset.global_mean if fill is None else float(fill)
        recset = []
        
        u = trainset.to_inner_uid(self.user_Id)
        user_items = set([j for (j, _) in trainset.ur[u]])
        recset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                         i in trainset.all_items() if
                         i not in user_items]
        return recset
    
        
    def recommend(self, nof_rec=5, verbose=True):
        '''
        Recommends Items from database based on user's ratings.
        
        Args:
            nof_rec (int) -- Number of recommendations to return. Default: 5
            verbose (bool) -- Whether to print the results or not. Default: True
            
        Returns:
            items(pandas.DataFrame) -- Df of items with top predicted rating for the logged in user.
        '''
        
        # check if logged in user has rated any items, and if not ask them to rate some in order to be able to get recommendtions
        self.user_login()
        while (self.user_Id not in self.nof_user_ratings.index):
            try: 
                if self.nof_user_ratings[self.user_Id] <= self.min_nof_ratings:
                    print('You have not rate enough items.')
            except:
                print('You have not rate any item.')
            rat_flag = self.input_y_n('Would you like to rate some items now (y/n)? ')
            if rat_flag in self.pos:
                self.new_ratings()
            else:
                print ('We cannot recommend Items to you.')
                return
        
        # Check if the ratings_df has changed since the last time we trained the model
        if self.ratings_changed:
            self.model_fit()
        
        # create set of user/items to use with surprise.algo and get predictions
        try:
            recset = self.build_recset(self.trainset)
        except:
            self.build_trainset()
            recset = self.build_recset(self.trainset)
        try:
            predictions = self.algorithm.test(recset)
        except:
            self.model_fit()
            predictions = self.algorithm.test(recset)
        
        # get the books with the top predicted rating and construct a pd.DataFrame of them and the ratings
        top_n = []
        for _, iid, _, est, _ in predictions:
            top_n.append((iid, est))
        top_n.sort(key=lambda x:x[1], reverse=True)        
        isbn, rating=[], []
        for i, r in top_n[:nof_rec]:
            isbn.append(i)
            rating.append(int(r))
        items = self.items_df.loc[self.items_df.isbn.isin(isbn)]
        items = pd.concat([items.reset_index(drop=True), pd.DataFrame({'rating':rating})], axis=1)
        
        # print the recommendations if asked to do so
        if verbose:
            print('For the User with ID %d we recommend: '%self.user_Id)
            for i, item in enumerate (items.iterrows()):
                #print('Book "',item[1], '" from "', item[2], '", (%f)'%item[5])
                print('{0}) "{1}" from {2}. ({3})'.format(i+1, item[1][1], item[1][2], int(top_n[i][1])))
        return items
        
            
    def get_dfs (self):
        '''
        Returns the DataFrames 
        '''
        return self.users_df, self.items_df, self.ratings_df    
    
    
    def save_dfs (self, to_save='all'):
        '''
        Save the selected DataFrames
        
        Args: 
            to_save (str) -- Items to save. One of ['all','users','items','ratings'].
        '''
        if to_save=='all' or to_save=='users':
            self.users_df.to_csv(os.path.join(self.dfs_path,'users_w_ex_ratings.csv'), sep=';', index=False)
        if to_save=='all' or to_save=='items':
            self.items_df.to_csv(os.path.join(self.dfs_path,'items_wo_duplicates.csv'), sep=';',index=False)
        if to_save=='all' or to_save=='ratings':
            self.ratings_df.to_csv(os.path.join(self.dfs_path,'ratings_expl.csv'),sep=';',index=False)
            
            
    def load_dfs(self):
        '''
        Load the DataFrames
        
        Returns:
            users_df -- pandas.DataFrame of users
            items_df -- pandas.DataFrame of items
            ratings_df -- pandas.DataFrame of ratings
        '''
        try:
            users_df = pd.read_csv(os.path.join(self.dfs_path,'users_w_ex_ratings.csv'), sep=';',encoding='latin-1',low_memory=False)        
            items_df = pd.read_csv(os.path.join(self.dfs_path,'items_wo_duplicates.csv'), sep=';',encoding='latin-1',low_memory=False)            
            ratings_df = pd.read_csv(os.path.join(self.dfs_path,'ratings_expl.csv'), sep=';',encoding='latin-1',low_memory=False)
        except:
            logging.error(('One or more of the files was not found in %s.\n Please make sure you have run '
                           '"BookCrossing data cleansing.ipynb" first.'), self.dfs_path)
            sys.exit(1)
        return users_df, items_df, ratings_df
    
    
if __name__=='__main__':
    rec = recommender()
    rec.main_menu()