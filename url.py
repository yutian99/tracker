import re

class Rule(object):
    def __init__(self, rule, list_limit, list_ignore, domain_limit, domain_ignore, exception) :
        try:
            self.rule = rule
            self.list_limit = list_limit
            self.list_ignore = list_ignore
            self.domain_limit = domain_limit
            self.domain_ignore = domain_ignore
            self.exception = exception
            if rule[0] == '^' :
                self.re_rule = re.compile(rule)
            else :
                self.re_rule = re.compile('.*' + rule)
            if list_limit : self.re_list_limit = re.compile('.*\\.(' + list_limit[1:] + ')')
            if list_ignore : self.re_list_ignore = re.compile('.*\\.(' + list_ignore[1:] + ')')
            if domain_limit : self.re_domain_limit = re.compile('.*(' + domain_limit[1:] + ')')
            if domain_ignore : self.re_domain_ignore = re.compile('.*(' + domain_ignore[1:] + ')')
            self.useful = 1
        except:
            #print(rule + '\n')
            self.useful = 0

    def Match(self, url) :
        if self.useful == 0 :
            return 0
        if str(self.re_rule.match(url)) == 'None' :
            return 0
        if (self.list_ignore and str(self.re_list_ignore.match(url)) != 'None') or (self.domain_ignore and str(self.re_domain_ignore.match(url)) != 'None') :
            return 0
        if (self.list_limit and str(self.re_list_limit.match(url)) == 'None') or (self.domain_limit and str(self.re_domain_limit.match(url)) == 'None') :
            return 0
        return 1

    def GetRule(self) :
        limit = ''
        if self.list_limit : limit = limit + self.list_limit
        if self.list_ignore : limit = limit + '~' + self.list_ignore
        if self.domain_limit : limit = limit + ' ' + self.domain_limit
        if self.domain_ignore : limit = limit + ' ~' + self.domain_ignore
        if self.exception == 1 : limit = limit + ' exception'
        return self.rule + '    ' + limit

urls = []
urls_num = 0
rules = []
rules_num = 0

dictionary = {}
dict_list = [[],]
dict_list_len = [0,]
dict_num = 1

match_count = 0
def GetDictValue(str_key) :
    if str_key in dictionary :
        return dictionary[str_key]
    return 0
rule_0 = 0
def DictInsertRule(rule_id) :
    parts = re.findall(r'[a-zA-Z0-9\-%_]+', rules[rule_id].rule)
    minn = 1000000
    minn_list_id = 0
    minn_part = 'null'
    #print(parts)
    global dict_num
    for part in parts :
        if part == 'http' or part == 'a-zA-Z0-9_' or part == '-' or part == '%' :
            continue
        list_id = GetDictValue(part)
        if list_id == 0 :
            dictionary[part] = dict_num
            dict_list.append([])
            dict_list_len.append(0)
            minn = 0
            minn_list_id = dict_num
            minn_part = part
            dict_num = dict_num + 1
            break
        #if len(dict_list[list_id]) != dict_list_len[list_id] :
        #    print('wrong')
        if dict_list_len[list_id] < minn :
            minn = dict_list_len[list_id]
            minn_list_id = list_id
            minn_part = part
    #print(minn_part + ' ' + str(minn_list_id) + ' ' + str(minn))
    #dict_list[minn_list_id].append(rule_id)
    #return
    if minn_list_id > 0 :
        dict_list[minn_list_id].append(rule_id)
        dict_list_len[minn_list_id] = dict_list_len[minn_list_id] + 1
        return
    global rule_0
    rule_0 = rule_0 + 1
    #print(str(rule_0) + ' ' + str(rule_id))
    if rules[rule_id].domain_limit == '' :
        #print('no domain_limit')
        return
    domains = rules[rule_id].domain_limit.split("|")
    for domain in domains :
        parts = re.findall(r'[a-zA-Z0-9\-%_]+', domain)
        minn = 1000000
        minn_list_id = 0
        minn_part = 'null'
        #print(parts)
        for part in parts :
            if part == 'http' or part == 'a-zA-Z0-9_' or part == '-' or part == '%' :
                continue
            list_id = GetDictValue(part)
            if list_id == 0 :
                #global dict_num
                dictionary[part] = dict_num
                dict_list.append([])
                dict_list_len.append(0)
                minn = 0
                minn_list_id = dict_num
                minn_part = part
                dict_num = dict_num + 1
                break
            if dict_list_len[list_id] < minn :
                minn = dict_list_len[list_id]
                minn_list_id = list_id
                minn_part = part
        #print(minn_part + ' ' + str(minn_list_id) + ' ' + str(minn))
        if minn_list_id > 0 :
            dict_list[minn_list_id].append(rule_id)
            dict_list_len[minn_list_id] = dict_list_len[minn_list_id] + 1
        #else :
        #    print('wrong')

option_list = ['stylesheet','xmlhttprequest','object-subrequest','subdocument','ping',
'websocket','webrtc','document','elemhide','generichide','genericblock','popup','other',
'third-party','first-party','sitekey','csp','match-case','collapse','donottrack','rewrite',]

part_list = ['js', 'ads', 'com', 'img', 'images', 'ad_', 'ad', 'net', 'php', 'common', 'gif', 'html', 'scripts', 'static', 'code', 'id', 'log', 'swf', 'assets', 'banner', 'view', 'google', 'jquery', 'iframe', 'png', 'amazonaws', 'event', 'cloudflare', 'cloudfront', 'libs', 'count', 'banners', 'facebook', 'adv', 'script', 'stats', 'min', 'yahoo', 'doubleclick', 'ajax', 'googleapis', 'ads_', '_ads', 'google-analytics', 'analytics', 'tracker', 'beacon', 'track', 'tracking', 'pixel', 'cgi', 'disqus', 'phncdn', 'connect', 'oas', '160x600', '468x60', '728x90', '120x600', 'cdnjs', 'cn', 'co', 'image', 'in', 'info', 'me', 'org', 'top', 'tv', 'us', 'a', 'api', 'c', 'cdn-cgi', 'p', 'content', 'data', 'i', 'include', 'web', 'main', 's', 'pic', 'public', 'show', 'qq', 'u', 'wp-content', '_ad', 'asp', 'ashx', '1', 'baidu', 'www', 'skin', 'modules', 'r', 'media', 't', 'https', 'news', 'htm', 'get', 'video', 'new', 'index', 'site', 'app', 'jpg', 'msn', 'sohu', '2', 'ad-', 'cms', 'g', 'z', 'b', 'action', 'widget', 'player', 'plugins', 'css', 'cdn', 'wp', 'v1', 'de', 'delivery', 'io', 'home', 'global', 'events', 'ws', 'files', 'json', 'affiliate', 'sina', 'go', 'javascript', 'aspx', 'pl', 'embed', 'imgur', 'win', 'pv', 'promo', 'service', 'popads', 'weather', 'pub', 'search', 'yimg', 'sinaimg', 'load', 'page', 'stat', 'click', 'url', 'statistics', 'lib', 'ads-', 'adx', 'amazon', 'adserver', 'adsense', 'advert', 'advertisement', 'sid', 'ui', 'criteo', 'pagead', 'exoclick', '300x250', 'googletagmanager', 'cgi-bin', 'type', 'translate', 'akamaihd', 'urchin', 'blank', 'optimizely', 'counter', 'gtm', 'ga', 'scorecardresearch', 'ping', 'referrer', 'dc', 'jsp', 'referer', 'uk', 'webstats', 'visit', 'pageview', 'report', 'apis', 'metrics', 'chartbeat', 'collect', 'aff', 'geo', 'logger', 'ng', 'imp', 'impression', 'includes', 'pix', 'outbrain', 'vidoza', 'flashx', 'atdmt', 'aol', 'mgid', 'gstatic', 'widgets', 'ioam', 'tx', 'media-imdb', 'rackcdn', 'twitter', 'vk', 'blob', 'thevideo', 'dailymotion', 'depositfiles', 'pbsrc', 'redtube', 'wired', 'platform', 'adobedtm', 'satelliteLib-', 'popunder', '-ads', 'ad2', 'adv_', 'adz', 'dfp', 'display', 'sponsors', 'adtech', '480x60', 'cdn77', 'streamplay', 'disquscdn', 'recaptcha', '2mdn', 'youjizz', 'amazon-adsystem', 'pleaseletmeadvertise', 'www-static', 'rdtcdn']
def ReadEasylist():
    global dict_num
    for part in part_list :
        dictionary[part] = dict_num
        dict_list.append([])
        dict_list_len.append(0)
        dict_num = dict_num + 1

    file_in = open('easylistall.txt', 'r')
    file_out = open('easylist_extract.txt', 'w')
    count = 0
    count_useful = 0
    line = file_in.readline().strip()
    while line:
        rule = line
        line = file_in.readline().strip()
        if rule == '' or rule[0] == '!' or rule[0] == '[' or rule.find('##') >= 0:
            continue
        exception = 0
        if rule.find('@@') == 0 :
            exception = 1
            rule = rule[2:]
        rule = rule.replace('.','\\.')
        rule = rule.replace('?','\\?')
        rule = rule.replace('*','.*')
        rule = rule.replace('^','[^a-zA-Z0-9_\-.%]')

        if rule.find('||') == 0 :
            rule = '^http://([^/]*\\.)?' + rule[2:]
        if rule[0] == '|' :
            rule = '^' + rule[1:]
        if rule[-1] =='|':
            rule = rule[: -1] + '$'
        rule=rule.replace('|','\\|')

        list_limit=''
        list_ignore=''
        domain_limit=''
        domain_ignore=''
        c = rule.find('$')
        if c >= 0 and c != len(rule) - 1:
            part = rule[c:]
            ignore = 0
            for option in option_list :
                if part.find(option) >= 0 :
                    ignore = 1
            if ignore == 1 :
                continue

            if part.find('~image')>=0: list_ignore=list_ignore + '|jpg|png|bmp|gif'
            elif part.find('image')>=0: list_limit=list_limit + '|jpg|png|bmp|gif'

            if part.find('~object')>=0: list_ignore=list_ignore + '|swf|jar'
            elif part.find('object')>=0: list_limit=list_limit + '|swf|jar'

            if part.find('~script')>=0: list_ignore=list_ignore + '|js|vbs'
            elif part.find('script')>=0: list_limit=list_limit + '|js|vbs'

            #if part.find('subdocument')>=0:r=r+'|.*'#9
            #if part.find('subrequest')>=0:r=r+'|.*'
            #if part.find('third-party')>=0:r=r+'|third-party'
            #if part.find('popup')>=0:r=r+'|.*'
            #if part.find('subdocument') >= 0 or part.find('subrequest') >= 0 or part.find('third-party') >= 0 or part.find('popup') >= 0:
            #    continue

            if part.find('domain')>=0:
                domain=part[part.find('domain=')+7:]
                if domain.find(',') >= 0: 
                    domain = domain[: domain.find(',')]
                if domain.find('~')>=0:
                    domain = domain.replace('~','')
                    domain_ignore = domain
                else:
                    domain_limit = domain
            rule = rule[: c]
            if c == 0 :
                rule = '.*'
        
        rule_struct = Rule(rule, list_limit, list_ignore, domain_limit, domain_ignore, exception)
        rules.append(rule_struct)
        file_out.write(rules[count].GetRule() + '\n')
        DictInsertRule(count)
        count = count + 1
        #print(count)
        if rules[count - 1].useful  == 1 :
            count_useful = count_useful + 1
        #if count > 10000 : break
    global rules_num
    rules_num = len(rules)
    print(rules_num)
    for rule in rules:
        print(rule.re_rule)
    #print(count_useful)
    #print(count)

def CheckTrackerForce(url_id, file_out) :
    flag_print = 1
    flag_tracker = 0
    global rules_num
    #print(rules_num)
    for rule_id in range(rules_num) :
        #print(i)
        if rules[rule_id].Match(urls[url_id]) == 1 :
            if flag_print == 1 :
                print('match ' + str(url_id) + ' ' + str(rule_id))
                file_out.write('match ' + str(url_id) + ' ' + str(rule_id) + '\n')
                file_out.write(urls[url_id] + '\n')
                file_out.write(rules[rule_id].GetRule() + '\n')
            if rules[rule_id].exception == 1 :
                return 0
            flag_tracker = 1
    if flag_tracker == 0 :
        return 0
    return 1

def CheckTracker(url_id, file_out) :
    flag_print = 1
    flag_tracker = 0
    parts = re.findall(r'[a-zA-Z0-9\-%_]+', urls[url_id])
    match_list_id = [0,]
    for part in parts :
        list_id = GetDictValue(part)
        if list_id != 0 :
            match_list_id.append(list_id)
    
    for list_id in match_list_id :
        for rule_id in dict_list[list_id] :
            if rules[rule_id].Match(urls[url_id]) == 1 :
                if flag_print == 1 :
                    print('match ' + str(url_id) + ' ' + str(rule_id))
                    #file_out.write('match ' + str(url_id) + ' ' + str(rule_id) + '\n')
                    #file_out.write(urls[url_id] + '\n')
                    #file_out.write(rules[rule_id].GetRule() + '\n')
                if rules[rule_id].exception == 1 :
                    return 0
                flag_tracker = 1
    if flag_tracker == 0 :
        return 0
    return 1

def CheckTrackerUrl(url) :
    flag_print = 1
    flag_tracker = 0
    parts = re.findall(r'[a-zA-Z0-9\-%_]+', url)
    match_list_id = [0,]
    for part in parts :
        list_id = GetDictValue(part)
        if list_id != 0 :
            match_list_id.append(list_id)
    
    global match_count
    for list_id in match_list_id :
        for rule_id in dict_list[list_id] :
            match_count = match_count + 1
            if rules[rule_id].Match(url) == 1 :
                #if flag_print == 1 :
                    #print('match ' + str(url_id) + ' ' + str(rule_id))
                    #file_out.write('match ' + str(url_id) + ' ' + str(rule_id) + '\n')
                    #file_out.write(urls[url_id] + '\n')
                    #file_out.write(rules[rule_id].GetRule() + '\n')
                if rules[rule_id].exception == 1 :
                    return False
                flag_tracker = 1
    if flag_tracker == 0 :
        return False
    return True

def ExactURL():
    file_in = open('zhengzhou201603232359.txt', 'r')
    file_out = open('url_extract.txt', 'w')
    match_out = open('match_result.txt', 'w')
    count = 0
    line = file_in.readline()
    while line:
        count = count + 1
        print(count)
        arr = line.split("|")
        for i in [59, 63]:
            url = arr[i]
            if len(url) == 0:
                continue
            file_out.write(url+'\n')
            urls.append(url)
            if CheckTrackerUrl(url) == True :
                match_out.write('match ' + str(count) + '\n')
        line = file_in.readline()
        #if count > 100: break
        #break
    global urls_num
    urls_num = len(urls)
    print(count)


def ExactURL2():
    match_out = open('../dataset/tracker_10000.txt', 'w')
    d_labels_f = open("../dataset/id_url_label_new_10000.txt", "r")
    count = 0
    for line in d_labels_f:
        line = line.strip()
        url = re.split(' ', line)[0]
        count = count + 1
        if count%10000==0:
            print("count:{}".format(count))
        if CheckTrackerUrl(url) == True:
            match_out.write(url + '\n')
    d_labels_f.close()
    match_out.close()


ReadEasylist()
#ExactURL2()

'''
test_part_list = []
value_sum = 0
for key in dictionary.keys():
    length = len(dict_list[dictionary[key]])
    if length > 5 :
        print(key + ' ' + str(length))
        test_part_list.append(key)
        value_sum = value_sum + length
print(len(dict_list[dictionary['com']]))
print(match_count)
print(test_part_list)
print(value_sum)
'''

'''
print(urls[14])
print(rules[63421].Match(urls[14]))
#list_limit = re.compile('(jpg|png|bmp|gif|js|vbs)')
list_limit = re.compile('^http://([^/]*\\.)?example.com/banner.gif')
if str(list_limit.match('http://gooddomain.example/analyze?http://example.com/banner.gif')) == 'None' :
    print('None')
else :
    print('yes')
'''

#MatchProcess()
#for i in range(100) :

#file_out = open('match2.txt', 'w')
#for url_id in range(urls_num) :
#    print(url_id)
#    CheckTracker(url_id, file_out)
