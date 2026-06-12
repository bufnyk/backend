from urllib.parse import urlparse

def globs(url: str, excluded: list=None):
    parsed_url = urlparse(url)
    result = []
    list_to_parse = ['blog', 'blogs']
    if excluded:
        for suf in excluded:
            if suf not in list_to_parse:
                list_to_parse.append(suf)

    for end in list_to_parse:
        result.append({"glob": f"https://{parsed_url.netloc}/{end}**"})
        result.append({"glob": f"https://{parsed_url.netloc}/{end}/**"})
        result.append({"glob": f"https://{parsed_url.netloc}/**/{end}"})
        result.append({"glob": f"https://{parsed_url.netloc}/**/{end}/**"})
    
    return result



