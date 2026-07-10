use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub fn read_wikipedia_extract(path: &Path) -> Option<(String, String)> {
    let content = fs::read_to_string(path).ok()?;
    let title = extract_json_field(&content, "title")?;
    let extract = extract_json_field(&content, "extract")?;
    if extract.is_empty() { return None; }
    Some((title, extract))
}

fn extract_json_field(json: &str, field: &str) -> Option<String> {
    let key_pattern = format!("\"{}\"", field);
    let pos = json.find(&key_pattern)?;
    let after = &json[pos + key_pattern.len()..];
    let colon = after.find(':')?;
    let after = &after[colon + 1..].trim_start();

    if after.starts_with('"') {
        let s = &after[1..];
        let mut result = String::new();
        let mut i = 0;
        let chars: Vec<char> = s.chars().collect();
        while i < chars.len() {
            match chars[i] {
                '\\' if i + 1 < chars.len() => {
                    match chars[i + 1] {
                        '"' => result.push('"'),
                        '\\' => result.push('\\'),
                        '/' => result.push('/'),
                        'n' => result.push('\n'),
                        'r' => result.push('\r'),
                        't' => result.push('\t'),
                        'u' => {},
                        c => result.push(c),
                    }
                    i += 2;
                }
                '"' => break,
                _ => { result.push(chars[i]); i += 1; }
            }
        }
        Some(result)
    } else {
        None
    }
}

pub fn build_category_map() -> HashMap<String, String> {
    let mut m = HashMap::new();

    let science = ["Newton","Darwin","DNA","Evolution","Calculus","Algebra","Carbon","Oxygen",
                   "Periodic_table","Philosophy","Photosynthesis","Gravity"];
    let tech = ["ChatGPT","Bitcoin","Internet_Explorer","Windows_XP","BlackBerry","Nokia",
                "MySpace","LimeWire","Vine_(service)","Second_Life","FarmVille",
                "Clubhouse_(app)","Google%2B","Adobe_Flash","NFT","Yahoo!","Flash_Player"];
    let entertainment = ["Barbie_(film)","Oppenheimer_(film)","Taylor_Swift",
                         "Academy_Awards","Grammy_Awards"];
    let politics = ["Democracy","Brexit","Black_Lives_Matter","Ukraine",
                    "Queen_Elizabeth_II","Trump"];
    let business = ["Elon_Musk","Tesla,_Inc.","GameStop_short_squeeze"];
    let sports = ["Olympic_Games","FIFA_World_Cup","Super_Bowl"];
    let culture = ["Christmas","Thanksgiving","Halloween","Beethoven","Shakespeare"];
    let health = ["COVID-19"];

    for s in science { m.insert(s.to_string(), "Science".to_string()); }
    for s in tech    { m.insert(s.to_string(), "Technology".to_string()); }
    for s in entertainment { m.insert(s.to_string(), "Entertainment".to_string()); }
    for s in politics { m.insert(s.to_string(), "Politics".to_string()); }
    for s in health   { m.insert(s.to_string(), "Health".to_string()); }
    for s in business { m.insert(s.to_string(), "Business".to_string()); }
    for s in sports   { m.insert(s.to_string(), "Sports".to_string()); }
    for s in culture  { m.insert(s.to_string(), "Culture".to_string()); }

    m
}
