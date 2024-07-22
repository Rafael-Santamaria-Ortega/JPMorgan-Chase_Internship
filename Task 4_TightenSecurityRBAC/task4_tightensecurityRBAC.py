"""
The goal of this coding activity is to design a system that limits the number of active roles that any given person has. 
A role gives the user access to some thing, whether it be a piece of data or an internal system. 
The system achieves this requirement by keeping track of the last k roles that a person has used. 
If a new role is used, the oldest role is removed if there are already k active roles for that person. 
Each role has a name and a message which contains details about its use by the person. 
You only need to store the last message for a role invocation.

Implement the constructor, get, and set methods of RolesCache. Each instance of the RolesCache corresponds to a single person.

Finally, fill out the runtime complexity for get and set and the overall space used. 
Use Big O notation, i.e. O(1), O(N), etc. 
For a refresher on Big O notation, please review https://danielmiessler.com/study/big-o-notation/.

"""

class RolesCache:
    def __init__(self, capacity):
        # Add any additional state you may need.
        self.capacity=capacity
        self.cache={}
        self.lru={}
        self.count=0

    def get(self, role):
        # Returns the message corresponding to the last invocation of that role, None if the role does not exist in the cache.
        if role in self.cache:
            self.lru[role]=self.count
            self.count+=1
        return self.cache.get(role,None)
        

    def set(self, role, message):
        # Adds the role and its corresponding message to the cache.
        # If the role already exists, only the message is updated. Otherwise, the oldest role is removed to make space.
        self.cache[role]=message
        if role in self.cache:
            self.lru[role]=self.count
            self.count+=1
            self.cache[role]=message
        else:
            if len(self.cache)>=capacity:
                oldest_role=None
                oldest_count=float('inf')
                for k,v in self.cache:
                    if self.lru[k]<oldest_count:
                        oldest_role=k
                        oldest_count=self.lru[k]
                self.cache.pop(oldest_role)
                self.lru.popitem(oldest_count)
                self.cache[role]=message

    def _complexity(self):
        return {
            'get': '0(1)',
            'set': '0(N)',
            'space': '0(N)'
        }