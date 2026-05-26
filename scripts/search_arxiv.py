import urllib.request, urllib.parse, xml.etree.ElementTree as ET

queries = [
    'all:"Measuring and Narrowing the Compositionality Gap"',
    'all:"Recitation-Augmented Language Models"',
    'all:"Chain-of-Thought Prompting Elicits Reasoning"',
    'all:"Self-Consistency Improves Chain of Thought"'
]

for q in queries:
    encoded_q = urllib.parse.quote(q)
    url = f'http://export.arxiv.org/api/query?search_query={encoded_q}&max_results=1'
    try:
        response = urllib.request.urlopen(url)
        data = response.read()
        root = ET.fromstring(data)
        entry = root.find('{http://www.w3.org/2005/Atom}entry')
        if entry is not None:
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.replace('\n', ' ')
            authors = [a.find('{http://www.w3.org/2005/Atom}name').text for a in entry.findall('{http://www.w3.org/2005/Atom}author')]
            published = entry.find('{http://www.w3.org/2005/Atom}published').text
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.replace('\n', ' ')
            print(f'Title: {title}\nAuthors: {authors}\nPublished: {published}\nAbstract: {summary[:500]}...\n')
    except Exception as e:
        print('Error:', e)
